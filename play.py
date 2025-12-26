"""
A3C 游戏画面渲染脚本
用于展示训练后的智能体玩游戏的过程

使用方法:
python play.py --env-name PongNoFrameskip-v4 --model-path saved_model.pt

按 Q 键退出游戏
"""
import argparse
import time
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

# 尝试导入pygame用于更好的渲染
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("提示: 安装pygame可获得更好的渲染效果: pip install pygame")

from envs import create_atari_env
from model import ActorCritic


def play_game(args):
    """
    使用训练好的模型玩游戏并渲染画面
    
    Args:
        args: 参数配置
    """
    # 创建环境 - 使用render_mode="human"来显示游戏画面
    try:
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStack as GymFrameStack
        
        # 创建可渲染的环境
        env_raw = gym.make(args.env_name, render_mode="human")
        
        # 应用与训练相同的预处理
        from envs import GymnasiumCompatWrapper, AtariRescale42x42, NormalizedEnv, FrameStack
        
        # 使用ALE内置的预处理
        env = AtariPreprocessing(env_raw, noop_max=30, frame_skip=4, 
                                  screen_size=84, terminal_on_life_loss=False,
                                  grayscale_obs=True, scale_obs=False)
        
        # 添加帧堆叠
        env = GymFrameStack(env, 4)
        
    except Exception as e:
        print(f"创建渲染环境失败: {e}")
        print("尝试使用备选方案...")
        
        # 备选方案：使用cv2渲染
        from envs import create_atari_env
        env = create_atari_env(args.env_name)
    
    # 加载模型
    model = ActorCritic(4, env.action_space)  # 4通道输入 (帧堆叠)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"加载模型: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    else:
        print("警告: 未指定模型路径或模型文件不存在，使用随机初始化的模型")
    
    model.eval()
    
    # 游戏统计
    total_episodes = 0
    total_reward = 0
    episode_rewards = []
    
    print("\n" + "="*50)
    print("A3C 游戏演示")
    print("="*50)
    print(f"环境: {args.env_name}")
    print(f"目标Episode数: {args.num_episodes}")
    print("按 Ctrl+C 退出")
    print("="*50 + "\n")
    
    try:
        for episode in range(args.num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # gymnasium返回(obs, info)
            
            # 转换状态格式
            if hasattr(state, '__array__'):
                state = np.array(state)
            
            # 确保状态形状正确 (4, 84, 84) 或 (4, 42, 42)
            if len(state.shape) == 3:
                if state.shape[0] != 4:  # 如果通道不在第一维
                    state = np.transpose(state, (2, 0, 1))
                # 缩放到42x42
                if state.shape[1] != 42:
                    from scipy.ndimage import zoom
                    state = zoom(state, (1, 42/state.shape[1], 42/state.shape[2]), order=1)
            
            state = torch.from_numpy(state.astype(np.float32) / 255.0)
            
            done = False
            episode_reward = 0
            episode_length = 0
            
            # 初始化LSTM隐藏状态
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            
            while not done:
                episode_length += 1
                
                with torch.no_grad():
                    value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
                
                prob = F.softmax(logit, dim=-1)
                
                # 选择动作 (贪婪策略)
                if args.stochastic:
                    action = prob.multinomial(num_samples=1).item()
                else:
                    action = prob.max(1, keepdim=True)[1].item()
                
                # 执行动作
                result = env.step(action)
                if len(result) == 5:  # gymnasium返回5个值
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:  # gym返回4个值
                    next_state, reward, done, info = result
                
                episode_reward += reward
                
                # 处理下一个状态
                if hasattr(next_state, '__array__'):
                    next_state = np.array(next_state)
                
                if len(next_state.shape) == 3:
                    if next_state.shape[0] != 4:
                        next_state = np.transpose(next_state, (2, 0, 1))
                    if next_state.shape[1] != 42:
                        from scipy.ndimage import zoom
                        next_state = zoom(next_state, (1, 42/next_state.shape[1], 42/next_state.shape[2]), order=1)
                
                state = torch.from_numpy(next_state.astype(np.float32) / 255.0)
                
                # 控制帧率
                time.sleep(1.0 / args.fps)
                
                if done:
                    break
            
            total_episodes += 1
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            
            print(f"Episode {total_episodes}: Reward = {episode_reward:.0f}, Length = {episode_length}, Avg = {total_reward/total_episodes:.1f}")
            
            # episode之间的短暂暂停
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n游戏被用户中断")
    finally:
        env.close()
    
    # 打印统计信息
    print("\n" + "="*50)
    print("游戏统计")
    print("="*50)
    print(f"总Episode数: {total_episodes}")
    print(f"平均奖励: {total_reward/max(1, total_episodes):.1f}")
    if episode_rewards:
        print(f"最高奖励: {max(episode_rewards):.0f}")
        print(f"最低奖励: {min(episode_rewards):.0f}")
    print("="*50)


def play_game_simple(args):
    """
    简化版本的游戏渲染，使用OpenCV显示
    """
    try:
        import cv2
    except ImportError:
        print("错误: 需要安装opencv-python: pip install opencv-python")
        return
    
    from envs import create_atari_env
    env = create_atari_env(args.env_name)
    
    # 加载模型
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"加载模型: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model.eval()
    
    print("\n" + "="*50)
    print("A3C 游戏演示 (OpenCV渲染)")
    print("="*50)
    print(f"环境: {args.env_name}")
    print("按 'q' 键退出")
    print("="*50 + "\n")
    
    total_episodes = 0
    total_reward = 0
    
    try:
        for episode in range(args.num_episodes):
            state = env.reset()
            state = torch.from_numpy(state)
            done = False
            episode_reward = 0
            
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            
            while not done:
                with torch.no_grad():
                    value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
                
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1, keepdim=True)[1].item()
                
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # 渲染状态 (第一帧)
                frame = state[0] if len(state.shape) == 3 else state
                frame = (frame * 255).astype(np.uint8)
                frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_NEAREST)
                
                # 添加信息文字
                cv2.putText(frame, f"Episode: {episode+1}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
                cv2.putText(frame, f"Reward: {episode_reward:.0f}", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
                
                cv2.imshow('A3C Pong', frame)
                
                if cv2.waitKey(int(1000/args.fps)) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
                
                state = torch.from_numpy(state)
            
            total_episodes += 1
            total_reward += episode_reward
            print(f"Episode {total_episodes}: Reward = {episode_reward:.0f}")
            
    except KeyboardInterrupt:
        print("\n游戏被用户中断")
    finally:
        cv2.destroyAllWindows()
        env.close()
    
    print(f"\n平均奖励: {total_reward/max(1, total_episodes):.1f}")


def record_video(args):
    """
    录制游戏视频
    """
    try:
        import cv2
    except ImportError:
        print("错误: 需要安装opencv-python: pip install opencv-python")
        return
    
    from envs import create_atari_env
    env = create_atari_env(args.env_name)
    
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model.eval()
    
    # 创建视频写入器
    os.makedirs('videos', exist_ok=True)
    video_path = os.path.join('videos', f'{args.env_name}_{int(time.time())}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, args.fps, (336, 336), isColor=False)
    
    print(f"录制视频到: {video_path}")
    
    frames = []
    
    for episode in range(args.num_episodes):
        state = env.reset()
        state = torch.from_numpy(state)
        done = False
        episode_reward = 0
        
        cx = torch.zeros(1, 256)
        hx = torch.zeros(1, 256)
        
        while not done:
            with torch.no_grad():
                value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1].item()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 处理帧
            frame = state[0] if len(state.shape) == 3 else state
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_NEAREST)
            
            out.write(frame)
            
            state = torch.from_numpy(state)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.0f}")
    
    out.release()
    env.close()
    print(f"\n视频已保存到: {video_path}")


def main():
    parser = argparse.ArgumentParser(description='A3C 游戏演示')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4',
                        help='游戏环境名称')
    parser.add_argument('--model-path', type=str, default=None,
                        help='训练好的模型路径')
    parser.add_argument('--num-episodes', type=int, default=5,
                        help='演示的episode数量')
    parser.add_argument('--fps', type=int, default=30,
                        help='帧率')
    parser.add_argument('--stochastic', action='store_true',
                        help='使用随机策略而非贪婪策略')
    parser.add_argument('--record', action='store_true',
                        help='录制视频而非实时显示')
    parser.add_argument('--simple', action='store_true',
                        help='使用简化的OpenCV渲染')
    
    args = parser.parse_args()
    
    if args.record:
        record_video(args)
    elif args.simple:
        play_game_simple(args)
    else:
        play_game(args)


if __name__ == '__main__':
    main()
