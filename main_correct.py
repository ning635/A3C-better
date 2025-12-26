"""
A3C主程序 - 严格按照论文实现
"""

import argparse
import torch
import torch.multiprocessing as mp
from model import ActorCritic
from shared_optim import SharedAdam, SharedRMSprop
from train_correct import train
try:
    import gymnasium as gym
except ImportError:
    import gym
from test import test
import os
import sys

parser = argparse.ArgumentParser(description='A3C')

parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=0.95,
                    help='lambda for GAE (default: 0.95)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='value to clip the grads (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps (default: 20)')
parser.add_argument('--n-workers', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--max-episode-length', type=int, default=10000,
                    help='maximum episode length (default: 10000)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on')
parser.add_argument('--use-trained', type=bool, default=False,
                    help='load trained model (default: False)')
parser.add_argument('--max-episodes', type=int, default=1000,
                    help='maximum training episodes (default: 1000)')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    mp.set_start_method('spawn')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    # 创建环境获取信息
    env = gym.make(args.env_name)
    print('Space dim: ', env.observation_space.shape)
    print('n. of actions: ', env.action_space.n)
    actions = env.action_space.n
    del env
    
    # 训练参数
    params = {
        'seed': args.seed,
        'env_name': args.env_name,
        'max_ep_length': args.max_episode_length,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'entropy_coef': args.entropy_coef,
        'value_coeff': args.value_loss_coef,
        'lr': args.lr,
        'n_process': args.n_workers,
        'max_grad_norm': args.max_grad_norm,
        'rollout_size': args.num_steps,
        'use_pre_trained': args.use_trained,
        'max_episodes': args.max_episodes if args.max_episodes > 0 else float('inf')
    }
    
    # 设置目标奖励
    if params['env_name'] == 'PongNoFrameskip-v4':
        params['mean_reward'] = 18.0
    elif params['env_name'] == 'BreakoutNoFrameskip-v4':
        params['mean_reward'] = 60.0
    else:
        print('No available env')
        sys.exit(1)
    
    # 网络参数
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
    
    if not params['use_pre_trained']:
        # 创建共享模型
        shared_model = ActorCritic(
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
        shared_model.share_memory()
        
        # 创建共享优化器
        # As in A3C paper, RMSprop performs better than Adam for asynchronous setting
        try:
            optimizer = SharedRMSprop(shared_model.parameters(), lr=params['lr'])
        except Exception:
            optimizer = SharedAdam(shared_model.parameters(), lr=params['lr'])
        optimizer.share_memory()
        
        # 多进程变量
        counter = mp.Value('i', 0)
        lock = mp.Lock()
        avg_ep = mp.Value('i', 0)
        scores = mp.Manager().list()
        scores_avg = mp.Manager().list()
        flag_exit = mp.Value('i', 0)
        
        print('----------- TRAINING INFO ------------')
        print(f'Learning rate: {params["lr"]}')
        print(f'Entropy coef: {params["entropy_coef"]}')
        print(f'GAE lambda: {params["gae_lambda"]}')
        print(f'Workers: {params["n_process"]}')
        print(f'Rollout size: {params["rollout_size"]}')
        print('--------------------------------------')
        
        # 启动训练进程
        processes = []
        for rank in range(params['n_process']):
            p = mp.Process(
                target=train,
                args=(rank, shared_model, params, optimizer, lock, counter, 
                      layers_, avg_ep, scores, scores_avg, flag_exit)
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        for p in processes:
            p.terminate()
        
        trained = True
    else:
        print('Loading trained model...')
        if params['env_name'] == 'PongNoFrameskip-v4':
            shared_model = torch.load('./saved_model/shared_model_pong.pt')
            trained = True
        elif params['env_name'] == 'BreakoutNoFrameskip-v4':
            shared_model = torch.load('./saved_model/shared_model_break.pt')
            trained = True
        else:
            print('No available trained model')
            sys.exit(1)
    
    # 测试
    if params['use_pre_trained'] or trained:
        test(params['n_process'], shared_model, params, params['max_ep_length'], layers_)
