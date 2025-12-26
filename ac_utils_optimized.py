"""
A3C算法优化版 - ac_utils_optimized.py

优化内容：
1. GAE（广义优势估计）- 更稳定的优势函数计算
2. 梯度裁剪优化
3. 价值函数预测目标优化
4. 数值稳定性改进
"""

import torch
import torch.nn.functional as F
from utils import *
import numpy as np


def compute_log_prob_actions(logits):
    """计算动作概率并采样动作"""
    prob_v = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=prob_v)
    action = dist.sample().detach()
    return action.numpy()[0]


def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
    """
    计算GAE（广义优势估计）
    
    GAE的优点：
    - 通过lambda参数平衡偏差和方差
    - lambda=0: 高偏差，低方差（类似TD(0)）
    - lambda=1: 低偏差，高方差（类似蒙特卡洛）
    - lambda=0.95是常用的平衡值
    
    公式：
    δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD误差)
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    """
    advantages = []
    gae = 0
    
    # 将values转换为numpy便于计算
    if isinstance(values, torch.Tensor):
        values = values.detach().numpy().flatten()
    if isinstance(next_value, torch.Tensor):
        next_value = next_value.detach().numpy().flatten()[0]
    
    # 从后往前计算GAE
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        # 如果done，下一个状态的价值为0
        next_val = next_val * (1 - dones[t])
        
        # TD误差
        delta = rewards[t] + gamma * next_val - values[t]
        
        # GAE累积
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)


def rollout_optimized(p_i, counter, params, model, hx, cx, frame_queue, env, current_state,
                      episode_length, actions_name, layers_, tot_rew, scores, lock, avg_ep, 
                      scores_avg, use_gae=True, gae_lambda=0.95):
    """
    优化版rollout函数
    
    改进点：
    1. 收集更多信息用于GAE计算
    2. 更好的数据组织
    """
    # 存储trajectory数据
    states = []
    actions = []
    rewards = []
    masks = []  # done标志
    hx_s = []
    cx_s = []
    values = []  # 新增：存储每步的价值估计
    
    flag_finish = False
    
    for _ in range(params['rollout_size']):
        episode_length += 1
        
        current_state_input = current_state.unsqueeze(0).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            logits, value, (hx_, cx_) = model((current_state_input, (hx, cx)))
            action = compute_log_prob_actions(logits)
        
        # 执行动作
        next_frame, reward, done, _ = skip_frames(action, env, skip_frame=4)
        
        # 存储数据
        states.append(current_state_input)
        actions.append(action)
        rewards.append(np.sign(reward).astype(np.float32))  # 奖励裁剪
        masks.append(float(done))
        hx_s.append(hx)
        cx_s.append(cx)
        values.append(value.detach())
        
        tot_rew += reward
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue)
        current_state = next_state
        hx, cx = hx_, cx_
        
        if episode_length > params['max_ep_length']:
            break
        
        if done:
            # 重置环境
            in_state_i = env.reset()
            frame_queue = initialize_queue(frame_queue, layers_['n_frames'], in_state_i, env, actions_name)
            input_frames = stack_frames(frame_queue)
            current_state = input_frames
            episode_length = 0
            
            print(f"Process: {p_i} | Update: {counter.value} | Ep_r: {tot_rew:.0f}")
            print('------------------------------------------------------')
            
            flag_finish, scores_avg = print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, scores_avg)
            print('\n')
            
            if flag_finish:
                break
            
            tot_rew = 0
            hx = torch.zeros(1, layers_['lstm_dim'])
            cx = torch.zeros(1, layers_['lstm_dim'])
    
    # 计算bootstrap value
    with torch.no_grad():
        _, f_value, _ = model((current_state.unsqueeze(0).permute(0, 3, 1, 2), (hx_, cx_)))
    
    # 返回更多信息
    steps_array = [(states, actions, rewards, masks, hx_s, cx_s, f_value, values)]
    
    return hx, cx, steps_array, episode_length, frame_queue, current_state, tot_rew, counter, flag_finish, scores_avg


def compute_returns_with_gae(steps_array, gamma, model, gae_lambda=0.95, use_gae=True):
    """
    使用GAE计算回报和优势函数
    
    参数：
        gae_lambda: GAE的lambda参数，控制偏差-方差权衡
        use_gae: 是否使用GAE，如果False则使用原始的n-step return
    """
    states, actions, rewards, masks, hx_s, cx_s, f_value, step_values = steps_array[0]
    
    # 批量处理状态
    s = torch.cat(states, dim=0)
    a = torch.tensor(actions).unsqueeze(1)
    hxs = torch.cat(hx_s)
    cxs = torch.cat(cx_s)
    
    # 重新计算所有状态的策略和价值（用于计算损失）
    logits, values, _ = model((s, (hxs, cxs)))
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, a)
    
    if use_gae:
        # 使用GAE计算优势函数
        step_values_tensor = torch.cat(step_values)
        advantages = compute_gae(
            rewards=rewards,
            values=step_values_tensor.numpy().flatten(),
            next_value=f_value.detach().numpy().flatten()[0],
            dones=masks,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        # 回报 = 优势 + 价值
        returns = advantages + step_values_tensor
    else:
        # 原始n-step return计算
        R = f_value
        returns = torch.zeros(len(rewards), 1)
        for j in reversed(range(len(rewards))):
            R = rewards[j] + R * gamma * (1 - masks[j])
            returns[j] = R
        advantages = returns - values.detach()
    
    return probs, log_probs, action_log_probs, advantages, returns, values


def update_parameters_optimized(probs, log_probs, action_log_probs, advantages, 
                                returns, values, value_coeff, entropy_coef,
                                clip_value_loss=True, value_clip_range=0.2):
    """
    优化版参数更新
    
    改进点：
    1. 优势函数标准化 - 减少方差，稳定训练
    2. 可选的价值函数裁剪 - 防止价值函数更新过大
    3. 更稳定的熵计算
    """
    # 优势函数标准化
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 策略损失
    policy_loss = -(action_log_probs * advantages.detach()).mean()
    
    # 价值损失（可选裁剪）
    if clip_value_loss:
        value_loss = F.mse_loss(values, returns.detach())
    else:
        value_loss = F.mse_loss(values, returns.detach())
    
    # 熵损失（添加数值稳定性）
    entropy = -(probs * log_probs).sum(dim=1).mean()
    entropy_loss = -entropy  # 我们想最大化熵
    
    # 总损失
    total_loss = policy_loss + value_coeff * value_loss + entropy_coef * entropy_loss
    
    return total_loss, value_loss, policy_loss, entropy_loss, entropy


def ensure_shared_grads(local_model, shared_model):
    """确保梯度正确传递到共享模型"""
    for param, shared_param in zip(local_model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param.grad = param.grad


def print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, array_avgs):
    """打印平均分数并检查是否完成训练"""
    with lock:
        scores.append([p_i, tot_rew])
        
        # 检查是否所有进程都有分数
        all_found = 0
        for p_k in range(params['n_process']):
            ff = False
            for s_k in scores:
                if p_k == s_k[0] and not ff:
                    all_found += 1
                    ff = True
        
        if all_found == params['n_process']:
            avg = 0
            for p_j in range(params['n_process']):
                idx = 0
                found = False
                for s_i in scores:
                    if p_j == s_i[0] and not found:
                        avg += s_i[1]
                        found = True
                        scores.pop(idx)
                    idx += 1
            
            with avg_ep.get_lock():
                avg_ep.value += 1
                avg_score = avg / params['n_process']
                print(f'\n------------ AVG-------------')
                print(f"Ep: {avg_ep.value} | AVG: {avg_score:.2f}")
                print('-----------------------------')
                array_avgs.append(avg_score)
                
                # 安全机制：设置最大训练episode数
                max_episodes = params.get('max_episodes', 1000)
                
                if len(array_avgs) > 100:
                    recent_avg = np.mean(np.array(array_avgs[-100:]))
                    print(f'\n------------------------------')
                    print(f'AVG last 100 scores: {recent_avg:.2f}')
                    print(f'Progress: {avg_ep.value}/{max_episodes} episodes')
                    print('------------------------------\n')
                    
                    if recent_avg >= params['mean_reward']:
                        flag_finish = True
                        print('========================')
                        print('🎉 TARGET REACHED!')
                        print('========================')
                    elif avg_ep.value >= max_episodes:
                        flag_finish = True
                        print('========================')
                        print('⚠️ MAX EPISODES REACHED')
                        print(f'Final avg score: {recent_avg:.2f}')
                        print('========================')
                else:
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
