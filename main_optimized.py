"""
A3C优化版主程序 - main_optimized.py

使用方法：
    python main_optimized.py --env-name PongNoFrameskip-v4 --n-workers 4 --use-gae True

优化点：
1. GAE（广义优势估计）
2. 优势函数标准化
3. 更好的日志和监控
"""

import argparse
import torch
import torch.multiprocessing as mp
from model import ActorCritic
from shared_optim import SharedAdam, SharedRMSprop
from train_optimized import train_optimized
import gym
from test import test
import os
import time
import sys

parser = argparse.ArgumentParser(description='A3C Optimized')

# 基本训练参数
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='value to clip the grads (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--rs', type=int, default=20,
                    help='rollout size before updating (default: 20)')
parser.add_argument('--n-workers', type=int, default=os.cpu_count(),
                    help='how many training processes to use (default: os cpus)')
parser.add_argument('--ep-length', type=int, default=int(4e10),
                    help='maximum episode length (default: 4e10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--opt', default='adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--use-trained', type=bool, default=False,
                    help='training A3C from scratch (default: False)')

# 优化相关参数
parser.add_argument('--use-gae', type=bool, default=True,
                    help='use Generalized Advantage Estimation (default: True)')
parser.add_argument('--gae-lambda', type=float, default=0.95,
                    help='GAE lambda parameter (default: 0.95)')


def print_config(params, layers_):
    """打印配置信息"""
    print('\n' + '='*60)
    print(' A3C OPTIMIZED CONFIGURATION')
    print('='*60)
    print(f" Environment:        {params['env_name']}")
    print(f" Workers:            {params['n_process']}")
    print(f" Learning Rate:      {params['lr']}")
    print(f" Gamma:              {params['gamma']}")
    print(f" Rollout Size:       {params['rollout_size']}")
    print(f" Use GAE:            {params.get('use_gae', True)}")
    print(f" GAE Lambda:         {params.get('gae_lambda', 0.95)}")
    print(f" Entropy Coef:       {params['entropy_coef']}")
    print(f" Value Loss Coef:    {params['value_coeff']}")
    print(f" Max Grad Norm:      {params['max_grad_norm']}")
    print(f" Target Reward:      {params['mean_reward']}")
    print('='*60 + '\n')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    mp.set_start_method('spawn')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    # 创建环境获取维度信息
    env_name = args.env_name
    env = gym.make(env_name)
    space = env.observation_space.shape
    actions = env.action_space.n
    print(f'Observation Space: {space}')
    print(f'Action Space: {actions}')
    del env
    
    # 创建保存目录
    os.makedirs('./saved_model', exist_ok=True)
    os.makedirs('./training_logs', exist_ok=True)
    
    # 训练参数
    params = {
        'seed': args.seed,
        'env_name': args.env_name,
        'max_ep_length': args.ep_length,
        'gamma': args.gamma,
        'entropy_coef': args.entropy_coef,
        'value_coeff': args.value_loss_coef,
        'lr': args.lr,
        'n_process': args.n_workers,
        'optimizer': args.opt,
        'max_grad_norm': args.max_grad_norm,
        'rollout_size': args.rs,
        'use_pre_trained': args.use_trained,
        'use_gae': args.use_gae,
        'gae_lambda': args.gae_lambda
    }
    
    # 设置目标奖励
    if params['env_name'] == 'PongNoFrameskip-v4':
        params['mean_reward'] = 18.0
    elif params['env_name'] == 'BreakoutNoFrameskip-v4':
        params['mean_reward'] = 60.0
    else:
        print('Environment not supported')
        sys.exit(1)
    
    # 网络结构参数
    layers_ = {
        'n_frames': 4,
        'hidden_dim1': 16,
        'kernel_size1': 8,
        'stride1': 4,
        'hidden_dim2': 32,
        'kernel_size2': 4,
        'stride2': 2,
        'fc1': 256,
        'lstm_dim': 256,
        'out_actor_dim': actions,
        'out_critic_dim': 1,
    }
    
    print_config(params, layers_)
    
    if not params['use_pre_trained']:
        # 创建共享模型
        shared_ac = ActorCritic(
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
        shared_ac.share_memory()
        
        # 创建共享优化器
        if params['optimizer'] == 'adam':
            optimizer = SharedAdam(shared_ac.parameters(), lr=params['lr'])
            optimizer.share_memory()
        elif params['optimizer'] == 'rmsprop':
            optimizer = SharedRMSprop(shared_ac.parameters(), lr=params['lr'])
            optimizer.share_memory()
        else:
            optimizer = None
        
        # 多进程共享变量
        counter_updates = mp.Value('i', 0)
        lock = mp.Lock()
        avg_ep = mp.Value('i', 0)
        scores = mp.Manager().list()
        scores_avg = mp.Manager().list()
        flag_exit = mp.Value('i', 0)
        trained = False
        
        n_processes = params['n_process']
        
        # 记录开始时间
        start_time = time.time()
        
        # 启动训练进程
        processes = []
        for p_i in range(n_processes):
            p = mp.Process(
                target=train_optimized,
                args=(p_i, shared_ac, params, optimizer, lock, counter_updates,
                      layers_, avg_ep, scores, scores_avg, flag_exit,
                      params['use_gae'], params['gae_lambda'])
            )
            p.start()
            processes.append(p)
        
        time.sleep(5)
        
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
        
        # 计算训练时间
        training_time = time.time() - start_time
        print(f'\n{"="*50}')
        print(f' TRAINING COMPLETE')
        print(f' Total Time: {training_time/60:.2f} minutes')
        print(f' Total Updates: {counter_updates.value}')
        print(f'{"="*50}\n')
        
        trained = True
    else:
        print('Loading pre-trained model...')
        if params['env_name'] == 'PongNoFrameskip-v4':
            model_path = './saved_model/shared_model_pong_optimized.pt'
            if not os.path.exists(model_path):
                model_path = './saved_model/shared_model_pong.pt'
            shared_ac = torch.load(model_path)
            trained = True
        elif params['env_name'] == 'BreakoutNoFrameskip-v4':
            model_path = './saved_model/shared_model_break_optimized.pt'
            if not os.path.exists(model_path):
                model_path = './saved_model/shared_model_break.pt'
            shared_ac = torch.load(model_path)
            trained = True
        else:
            print('No trained model available')
            sys.exit(1)
    
    # 测试
    if params['use_pre_trained'] or trained:
        test(params['n_process'], shared_ac, params, params['max_ep_length'], layers_)
