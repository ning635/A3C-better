import torch
import torch.nn.functional as F
from utils import *

def compute_log_prob_actions(logits):
    prob_v = F.softmax(logits, dim=-1)
    log_prob_v = F.log_softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=prob_v)
    action = dist.sample()
    action_log_prob = log_prob_v[0, action.item()]
    return action.item(), prob_v, log_prob_v, action_log_prob


def rollout(p_i, counter, params, model, hx, cx, frame_queue, env, current_state, episode_length, actions_name, layers_, tot_rew, scores, lock, avg_ep, scores_avg):
    
    #empty lists
    values = []
    log_probs = []
    rewards = []
    entropies = []
    dones = []  # 添加done标志列表
    
    flag_finish = False
    done = False  # 初始化done
    
    for _ in range(params['rollout_size']):
        episode_length +=1
        
        current_state = current_state.unsqueeze(0).permute(0,3,1,2)
        
        # 计算logits, value和LSTM状态
        logits, value, (hx, cx) = model((current_state, (hx, cx)))
        
        # 计算概率和采样动作
        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        
        action = prob.multinomial(num_samples=1).detach()
        action_log_prob = log_prob.gather(1, action)
        
        # 执行动作
        next_frame, reward, done, _ = skip_frames(action.item(), env, skip_frame=4)
        
        # 存储
        values.append(value)
        log_probs.append(action_log_prob)
        rewards.append(np.sign(reward))  # 奖励裁剪到[-1, 1]
        entropies.append(entropy)
        dones.append(done)  # 保存done标志
        
        tot_rew += reward
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue)
        current_state = next_state
        
        if episode_length > params['max_ep_length']:
            break
        
        if done:
            #reset env
            in_state_i = env.reset()
            frame_queue = initialize_queue(frame_queue, layers_['n_frames'], in_state_i, env, actions_name)
            input_frames = stack_frames(frame_queue)
            current_state = input_frames
            episode_length = 0
            print(
                "Process: ", p_i,
                "Update:", counter.value,
                "| Ep_r: %.0f" % tot_rew,
            )
            print('------------------------------------------------------')
            flag_finish, scores_avg = print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, scores_avg)                        
            print('\n')
            if flag_finish == True:
                break
            
            tot_rew = 0
            hx = torch.zeros(1, layers_['lstm_dim'])
            cx = torch.zeros(1, layers_['lstm_dim'])
        
    # bootstrapping - 如果最后一步episode结束了，R=0；否则用网络估计
    if done:
        R = torch.zeros(1, 1)
    else:
        with torch.no_grad():
            _, R, _ = model((current_state.unsqueeze(0).permute(0,3,1,2), (hx, cx)))
    
    steps_array = (values, log_probs, rewards, entropies, dones, R)
    
    return hx, cx, steps_array, episode_length, frame_queue, current_state, tot_rew, counter, flag_finish, scores_avg


def compute_returns(steps_array, gamma, model):
    """计算returns和losses - 正确处理episode边界"""
    values, log_probs, rewards, entropies, dones, R = steps_array
    
    R = R.detach()  # bootstrap value
    
    # 从后往前计算returns，正确处理episode边界
    returns = []
    for i in reversed(range(len(rewards))):
        # 如果这一步episode结束了，R重置为0再计算
        if dones[i]:
            R = torch.zeros(1, 1)
        R = rewards[i] + gamma * R
        returns.insert(0, R)
    
    # 计算advantages
    advantages = []
    for i in range(len(returns)):
        adv = returns[i] - values[i].detach()
        advantages.append(adv)
    
    # 标准化advantages（非常重要！）
    if len(advantages) > 1:
        advantages_tensor = torch.cat(advantages)
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std() + 1e-8
    else:
        adv_mean = 0
        adv_std = 1
    
    # 计算losses
    policy_loss = 0
    value_loss = 0
    entropy_sum = 0
    
    for i in range(len(rewards)):
        # 标准化的advantage
        normalized_adv = (advantages[i] - adv_mean) / adv_std
        
        # Policy loss
        policy_loss = policy_loss - log_probs[i] * normalized_adv.detach()
        
        # Value loss
        value_loss = value_loss + 0.5 * (returns[i].detach() - values[i]).pow(2)
        
        # Entropy
        entropy_sum = entropy_sum + entropies[i]
    
    # entropy_loss: 负熵，加到loss中相当于鼓励探索
    entropy_loss = -entropy_sum
    
    return policy_loss, value_loss, entropy_loss
    
    
def ensure_shared_grads(local_model, shared_model):
    """将本地模型的梯度复制到共享模型"""
    for param, shared_param in zip(local_model.parameters(), shared_model.parameters()):
        if param.grad is not None:
            if shared_param.grad is None:
                shared_param.grad = param.grad.clone()
            else:
                shared_param.grad += param.grad 
    

def update_parameters(probs, log_probs, action_log_probs, advantages, returns, values, value_coeff, entropy_coef):
    # 标准化优势函数 - 减少方差，稳定训练
    adv_normalized = advantages.detach()  # 对于policy loss，advantage不需要梯度
    if adv_normalized.numel() > 1:
        adv_normalized = (adv_normalized - adv_normalized.mean()) / (adv_normalized.std() + 1e-8)
    
    #policy loss (使用标准化的advantage，并detach防止梯度流向value网络)
    policy_loss = -(action_log_probs * adv_normalized).mean() 
    
    #value loss (returns是目标，没有梯度；values有梯度)
    value_loss = torch.nn.functional.mse_loss(values, returns)
    
    #entropy loss (负的熵，因为我们想最大化熵来保持探索)
    entropy_loss = (probs * log_probs).sum(dim=1).mean()
    
    a3c_loss = policy_loss + value_coeff * value_loss + entropy_coef * entropy_loss
    
    return a3c_loss, value_loss, policy_loss, entropy_loss
    
def print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, array_avgs):
    print('\n')
    with lock:
        scores.append([p_i, tot_rew])
        #print('scores', scores)
        all_found = 0
        #check if all process present
        for p_k in range(0, params['n_process']):
            ff = False
            for s_k in scores:
                if p_k == s_k[0] and ff==False:
                    all_found+=1
                    ff = True
                
        if all_found == params['n_process']:
            avg = 0
            for p_j in range(0, params['n_process']):
                idx = 0
                found = False
                for s_i in scores:
                    if p_j == s_i[0] and found==False:
                        avg += s_i[1]
                        found=True
                        scores.pop(idx)
                    idx+=1
                    
            with avg_ep.get_lock():
                avg_ep.value +=1
                print('\n')
                print('------------ AVG-------------')
                print(f"Ep: {avg_ep.value} | AVG: {avg/params['n_process']}")
                print('-----------------------------')
                array_avgs.append(avg/params['n_process'])
                
                # 安全机制：设置最大训练episode数（默认1000）
                max_episodes = params.get('max_episodes', 1000)
                
                if len(array_avgs)>100:
                    avg = np.mean(np.array(array_avgs[-100:]))
                    print('\n')
                    print('------------------------------')
                    print(f'AVG last 100 scores: {avg:.2f}')
                    print(f'Progress: {avg_ep.value}/{max_episodes} episodes')
                    print('------------------------------')
                    print('\n')
                    if avg >= params['mean_reward']:
                        flag_finish = True
                        print('========================')
                        print('🎉 TARGET REACHED!')
                        print('========================')
                    elif avg_ep.value >= max_episodes:
                        flag_finish = True
                        print('========================')
                        print('⚠️ MAX EPISODES REACHED')
                        print(f'Final avg score: {avg:.2f}')
                        print('========================')
                else:
                    # 即使没到100个episode，也检查是否超过最大限制
                    if avg_ep.value >= max_episodes:
                        flag_finish = True
                        print('========================')
                        print('⚠️ MAX EPISODES REACHED')
                        print('========================')
                    else:
                        flag_finish = False
        else:
            print('Not enough process completed to compute AVG...')
            flag_finish = False
        
        return flag_finish, array_avgs