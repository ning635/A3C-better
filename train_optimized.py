"""
优化版训练脚本 - train_optimized.py

主要优化：
1. 使用GAE（广义优势估计）
2. 学习率调度
3. 优势函数标准化
4. 更好的日志记录
"""

import torch
from model import ActorCritic
from utils import *
from ac_utils_optimized import *
from test import test
import time
import gym
import numpy as np
from collections import deque


def get_lr_scheduler(optimizer, total_updates, warmup_updates=1000, min_lr=1e-6):
    """
    学习率调度器
    
    策略：
    1. Warmup阶段：线性增加学习率
    2. 之后：线性衰减到最小值
    """
    def lr_lambda(current_update):
        if current_update < warmup_updates:
            # Warmup: 线性增加
            return current_update / warmup_updates
        else:
            # 线性衰减
            progress = (current_update - warmup_updates) / max(1, total_updates - warmup_updates)
            return max(min_lr / optimizer.defaults['lr'], 1.0 - progress)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_optimized(p_i, shared_model, p, optimizer, lock, counter, lys, avg_ep, 
                    scores, scores_avg, flag_exit, use_gae=True, gae_lambda=0.95):
    """
    优化版训练函数
    
    新增参数：
        use_gae: 是否使用GAE
        gae_lambda: GAE的lambda参数
    """
    params = p.copy()
    layers_ = lys.copy()
    
    seed = params['seed']
    torch.manual_seed(seed + p_i)
    np.random.seed(seed + p_i)
    
    env = gym.make(params['env_name'])
    actions_name = env.unwrapped.get_action_meanings()
    
    if p_i == 0:
        print('\n' + '='*50)
        print(' OPTIMIZED A3C TRAINING')
        print(f' GAE: {use_gae}, Lambda: {gae_lambda}')
        print('='*50 + '\n')
    
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
    
    if optimizer is None:
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=params['lr'])
    
    model.train()
    
    # 初始化环境
    queue = deque(maxlen=4)
    in_state_i = env.reset(seed=(seed + p_i))
    frame_queue = initialize_queue(queue, layers_['n_frames'], in_state_i, env, actions_name)
    input_frames = stack_frames(frame_queue)
    current_state = input_frames
    
    episode_length = 0
    tot_rew = 0
    
    # LSTM隐藏状态初始化
    hx = torch.zeros(1, layers_['lstm_dim'])
    cx = torch.zeros(1, layers_['lstm_dim'])
    
    # 记录训练统计
    local_updates = 0
    
    while True:
        # 检查退出标志
        if flag_exit.value == 1:
            print(f"Process {p_i} terminating...")
            break
        
        optimizer.zero_grad()
        
        # 同步本地模型与共享模型
        model.load_state_dict(shared_model.state_dict())
        
        # Rollout：收集经验
        (hx, cx, steps_array, episode_length, frame_queue, current_state, 
         tot_rew, counter, flag_finish, scores_avg) = rollout_optimized(
            p_i, counter, params, model, hx, cx, frame_queue, env, current_state,
            episode_length, actions_name, layers_, tot_rew, scores, lock, avg_ep, 
            scores_avg, use_gae=use_gae, gae_lambda=gae_lambda
        )
        
        if flag_finish:
            print('Saving model...')
            if params['env_name'] == 'PongNoFrameskip-v4':
                torch.save(shared_model, './saved_model/shared_model_pong_optimized.pt')
            elif params['env_name'] == 'BreakoutNoFrameskip-v4':
                torch.save(shared_model, './saved_model/shared_model_break_optimized.pt')
            
            plot_avg_scores(scores_avg, 'A3C Optimized - Average Scores')
            
            with flag_exit.get_lock():
                flag_exit.value = 1
            break
        
        # 计算回报和优势（使用GAE）
        (probs, log_probs, action_log_probs, advantages, 
         returns, values) = compute_returns_with_gae(
            steps_array, params['gamma'], model, 
            gae_lambda=gae_lambda, use_gae=use_gae
        )
        
        # 计算损失并更新参数
        (a3c_loss, value_loss, policy_loss, 
         entropy_loss, entropy) = update_parameters_optimized(
            probs, log_probs, action_log_probs, advantages, returns, values,
            params['value_coeff'], params['entropy_coef']
        )
        
        # 反向传播
        a3c_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
        
        # 同步梯度到共享模型
        ensure_shared_grads(model, shared_model)
        
        # 优化器更新
        optimizer.step()
        
        local_updates += 1
        
        # 定期打印训练信息
        if counter.value % 100 == 0:
            print(f'\n--- Process {p_i} | Update {counter.value} ---')
            print(f'Policy Loss:  {policy_loss.item():.4f}')
            print(f'Value Loss:   {value_loss.item():.4f}')
            print(f'Entropy:      {entropy.item():.4f}')
            print(f'Total Loss:   {a3c_loss.item():.4f}')
            print('-' * 40)
        
        with counter.get_lock():
            counter.value += 1
