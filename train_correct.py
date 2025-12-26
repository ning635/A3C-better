"""
A3C训练代码 - 严格按照论文实现
参考: https://github.com/ikostrikov/pytorch-a3c
"""

import torch
import torch.nn.functional as F
from model import ActorCritic
from utils import frame_preprocessing, stack_frames, initialize_queue, skip_frames, plot_avg_scores
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
from collections import deque


def ensure_shared_grads(model, shared_model):
    """将本地模型的梯度复制到共享模型 - 这是A3C的关键
    
    注意：使用_grad而不是grad，因为grad是只读属性
    """
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return  # 如果已经有梯度，说明另一个worker已经更新过了
        shared_param._grad = param.grad


def train(rank, shared_model, params, optimizer, lock, counter, layers_, avg_ep, scores, scores_avg, flag_exit):
    """
    A3C worker训练函数
    """
    torch.manual_seed(params['seed'] + rank)
    np.random.seed(params['seed'] + rank)
    
    # 创建环境
    # Prefer Gymnasium with canonical wrappers
    try:
        from atari_env import make_env
        env = make_env(params['env_name'], seed=(params['seed'] + rank), frame_stack=layers_['n_frames'])
    except Exception:
        env = gym.make(params['env_name'])
    actions_name = getattr(env.unwrapped, 'get_action_meanings', lambda: [])()
    
    print(f' ----- TRAIN PHASE (Worker {rank}) -----')
    
    # 创建本地模型
    model = ActorCritic(
        input_shape=layers_['n_frames'],
        layer1=layers_['hidden_dim1'],
        kernel_size1=layers_['kernel_size1'],
        stride1=layers_['stride1'],
        layer2=layers_['hidden_dim2'],
        kernel_size2=layers_['kernel_size2'],
        stride2=layers_['stride2'],
        fc1_dim=layers_['fc1'],
        lstm_dim=layers_['lstm_dim'],
        out_actor_dim=layers_['out_actor_dim'],
        out_critic_dim=layers_['out_critic_dim']
    )
    model.train()
    
    # 初始化环境
    # Initial state
    reset_out = env.reset(seed=(params['seed'] + rank))
    if isinstance(reset_out, tuple):
        obs, _ = reset_out
    else:
        obs = reset_out
    # Ensure HWC uint8
    import numpy as np
    if obs.dtype != np.uint8:
        obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)
    state = torch.from_numpy(obs)
    
    done = True
    episode_length = 0
    episode_reward = 0
    
    while True:
        # 检查是否应该退出
        if flag_exit.value == 1:
            print(f"Worker {rank} terminating...")
            break
        
        # 同步本地模型和共享模型
        model.load_state_dict(shared_model.state_dict())
        
        # 重置LSTM状态
        if done:
            hx = torch.zeros(1, layers_['lstm_dim'])
            cx = torch.zeros(1, layers_['lstm_dim'])
        else:
            hx = hx.detach()
            cx = cx.detach()
        
        # 收集经验的列表
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        # 收集 n 步经验
        for step in range(params['rollout_size']):
            episode_length += 1
            
            # 前向传播 (需要梯度!)
            state_tensor = state.unsqueeze(0).permute(0, 3, 1, 2)
            logits, value, (hx, cx) = model((state_tensor, (hx, cx)))
            
            # 计算概率分布
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            
            # 计算熵 (正值)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            
            # 采样动作
            action = prob.multinomial(num_samples=1).detach()
            log_prob_action = log_prob.gather(1, action)
            
            # 执行动作
            step_out = env.step(action.item())
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_out
            
            # 限制最大episode长度
            done = done or episode_length >= params['max_ep_length']
            
            # Clip reward到[-1, 1]
            reward_clipped = max(min(reward, 1), -1)
            
            # 记录
            episode_reward += reward
            
            # 更新counter
            with counter.get_lock():
                counter.value += 1
            
            # 如果episode结束
            if done:
                # 打印信息
                print(f"Process: {rank} Update: {counter.value} | Ep_r: {episode_reward:.0f}")
                print('------------------------------------------------------')
                
                # 更新平均分
                flag_finish, scores_avg_new = print_avg(scores, rank, episode_reward, lock, avg_ep, params, False, scores_avg)
                scores_avg = scores_avg_new
                
                if flag_finish:
                    # 保存模型
                    print('Save Model...')
                    if params['env_name'] == 'PongNoFrameskip-v4':
                        torch.save(shared_model, './saved_model/shared_model_pong.pt')
                    elif params['env_name'] == 'BreakoutNoFrameskip-v4':
                        torch.save(shared_model, './saved_model/shared_model_break.pt')
                    plot_avg_scores(scores_avg, 'Plot AVG Scores')
                    
                    with flag_exit.get_lock():
                        flag_exit.value = 1
                    return
                
                # 重置
                episode_length = 0
                episode_reward = 0
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                if obs.dtype != np.uint8:
                    obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)
                state = torch.from_numpy(obs)
            else:
                # 更新状态
                obs = next_obs
                if obs.dtype != np.uint8:
                    obs = (obs * 255).astype(np.uint8) if obs.max() <= 1.0 else obs.astype(np.uint8)
                state = torch.from_numpy(obs)
            
            # 保存value, log_prob, reward
            values.append(value)
            log_probs.append(log_prob_action)
            rewards.append(reward_clipped)
            
            if done:
                break
        
        # 计算bootstrap value R
        R = torch.zeros(1, 1)
        if not done:
            state_tensor = state.unsqueeze(0).permute(0, 3, 1, 2)
            with torch.no_grad():
                _, value, _ = model((state_tensor, (hx, cx)))
            R = value.detach()
        
        values.append(R)
        
        # 计算损失
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        
        # 从后往前计算 (这是A3C的核心!)
        for i in reversed(range(len(rewards))):
            R = params['gamma'] * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            # GAE (Generalized Advantage Estimation)
            delta_t = rewards[i] + params['gamma'] * values[i + 1] - values[i]
            gae = gae * params['gamma'] * params.get('gae_lambda', 0.95) + delta_t
            
            # Policy loss = -log_prob * advantage - entropy_coef * entropy
            # 注意: 熵是正的，我们要最大化熵，所以用减号
            policy_loss = policy_loss - log_probs[i] * gae.detach() - params['entropy_coef'] * entropies[i]
        
        # 反向传播
        optimizer.zero_grad()
        total_loss = policy_loss + params['value_coeff'] * value_loss
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
        
        # 复制梯度到共享模型
        ensure_shared_grads(model, shared_model)
        
        # 更新共享模型
        optimizer.step()
        
        # 打印训练信息
        if counter.value % 500 == 0:
            print(f'Worker: {rank} | Update: {counter.value}')
            print(f'  Policy Loss: {policy_loss.item():.4f}')
            print(f'  Value Loss: {value_loss.item():.4f}')
            print(f'  Total Loss: {total_loss.item():.4f}')
            print('------------------------------------------------------')


def print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, array_avgs):
    """计算并打印平均分数"""
    with lock:
        scores.append([p_i, tot_rew])
        
        # 检查是否所有进程都完成了一个episode
        all_found = 0
        for p_k in range(params['n_process']):
            for s_k in scores:
                if p_k == s_k[0]:
                    all_found += 1
                    break
        
        if all_found == params['n_process']:
            avg = 0
            for p_j in range(params['n_process']):
                for idx, s_i in enumerate(list(scores)):
                    if p_j == s_i[0]:
                        avg += s_i[1]
                        scores.remove(s_i)
                        break
            
            with avg_ep.get_lock():
                avg_ep.value += 1
                avg_score = avg / params['n_process']
                print('\n------------ AVG -------------')
                print(f"Ep: {avg_ep.value} | AVG: {avg_score:.2f}")
                print('------------------------------\n')
                array_avgs.append(avg_score)
                
                # 检查是否达到目标
                max_episodes = params.get('max_episodes', 1000)
                
                if len(array_avgs) >= 100:
                    recent_avg = np.mean(np.array(array_avgs[-100:]))
                    print(f'AVG last 100 scores: {recent_avg:.2f}')
                    print(f'Progress: {avg_ep.value}/{max_episodes} episodes\n')
                    
                    if recent_avg >= params['mean_reward']:
                        flag_finish = True
                        print('========================')
                        print('🎉 TARGET REACHED!')
                        print('========================')
                elif avg_ep.value >= max_episodes:
                    flag_finish = True
                    print('========================')
                    print('⚠️ MAX EPISODES REACHED')
                    print('========================')
    
    return flag_finish, array_avgs
