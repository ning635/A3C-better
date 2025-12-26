"""
A3C 游戏画面演示 - 支持实时渲染和GIF录制
使用gymnasium原生渲染或OpenCV显示

使用方法:
    python show_game.py                    # 实时渲染游戏画面
    python show_game.py --record           # 录制GIF
    python show_game.py --model model.pt   # 使用训练好的模型
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F


def create_render_env(env_name):
    """创建可渲染的游戏环境"""
    import gymnasium as gym
    
    # 创建带渲染的环境
    env = gym.make(env_name, render_mode="human")
    print(f"Created environment: {env_name}")
    print(f"Action space: {env.action_space}")
    return env


def create_recording_env(env_name):
    """创建用于录制的环境"""
    import gymnasium as gym
    
    env = gym.make(env_name, render_mode="rgb_array")
    return env


def load_trained_model(model_path, num_actions=6):
    """加载训练好的模型"""
    from model import ActorCritic
    import gymnasium as gym
    
    # 创建临时环境获取action_space
    temp_env = gym.make("PongNoFrameskip-v4")
    model = ActorCritic(4, temp_env.action_space)
    temp_env.close()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if hasattr(state_dict, 'state_dict'):
                model.load_state_dict(state_dict.state_dict())
            elif isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
            else:
                model = state_dict
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using random model")
    else:
        print("No model specified, using random actions")
        return None
    
    model.eval()
    return model


def preprocess_frame(frame):
    """预处理单帧画面"""
    import cv2
    
    # 裁剪和缩放
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (42, 42))
    
    # 转为灰度并归一化
    if len(frame.shape) == 3:
        frame = np.mean(frame, axis=2)
    frame = frame.astype(np.float32) / 255.0
    
    return frame


class FrameStacker:
    """帧堆叠器"""
    def __init__(self, k=4):
        self.k = k
        self.frames = []
    
    def reset(self, frame):
        processed = preprocess_frame(frame)
        self.frames = [processed] * self.k
        return self._get_stacked()
    
    def add(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        if len(self.frames) > self.k:
            self.frames.pop(0)
        return self._get_stacked()
    
    def _get_stacked(self):
        stacked = np.stack(self.frames, axis=0)
        return stacked.astype(np.float32)


def play_with_render(args):
    """实时渲染游戏画面"""
    print("\n" + "="*60)
    print(" A3C PONG DEMO - Real-time Rendering")
    print("="*60)
    print(" Close the game window to exit")
    print("="*60 + "\n")
    
    env = create_render_env(args.env_name)
    model = load_trained_model(args.model) if args.model else None
    
    frame_stacker = FrameStacker(k=4)
    hx = torch.zeros(1, 256)
    cx = torch.zeros(1, 256)
    
    total_rewards = []
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        state = frame_stacker.reset(obs)
        
        episode_reward = 0
        done = False
        step = 0
        
        # 重置LSTM状态
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        while not done:
            step += 1
            
            # 选择动作
            if model is not None:
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    value, logit, (hx, cx) = model((state_tensor, (hx, cx)))
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1)[1].item()
            else:
                action = env.action_space.sample()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # 更新状态
            state = frame_stacker.add(obs)
            
            # 控制帧率
            time.sleep(0.01)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.0f}, Steps = {step}")
    
    env.close()
    
    # 打印统计
    if total_rewards:
        print("\n" + "="*60)
        print(" Statistics")
        print("="*60)
        print(f" Episodes: {len(total_rewards)}")
        print(f" Average: {np.mean(total_rewards):.1f}")
        print(f" Best: {max(total_rewards):.0f}")
        print(f" Worst: {min(total_rewards):.0f}")
        print("="*60)


def record_gif(args):
    """录制游戏GIF"""
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio")
        return
    
    print("\n" + "="*60)
    print(" A3C PONG DEMO - Recording GIF")
    print("="*60 + "\n")
    
    env = create_recording_env(args.env_name)
    model = load_trained_model(args.model) if args.model else None
    
    frame_stacker = FrameStacker(k=4)
    
    os.makedirs('videos', exist_ok=True)
    
    for episode in range(args.episodes):
        frames = []
        obs, info = env.reset()
        state = frame_stacker.reset(obs)
        
        # 保存初始帧
        frames.append(env.render())
        
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        episode_reward = 0
        done = False
        
        while not done:
            if model is not None:
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    value, logit, (hx, cx) = model((state_tensor, (hx, cx)))
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1)[1].item()
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = frame_stacker.add(obs)
            
            # 保存帧
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # 保存GIF
        gif_path = f'videos/pong_ep{episode+1}_reward{int(episode_reward)}.gif'
        
        # 降采样以减小文件大小
        frames_sampled = frames[::2]  # 每2帧取1帧
        imageio.mimsave(gif_path, frames_sampled, fps=30)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.0f} -> Saved: {gif_path}")
    
    env.close()
    print(f"\nGIFs saved to videos/ folder")


def main():
    parser = argparse.ArgumentParser(description='A3C Game Demo with Rendering')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4',
                        help='Game environment')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes')
    parser.add_argument('--record', action='store_true',
                        help='Record GIF instead of real-time rendering')
    
    args = parser.parse_args()
    
    try:
        if args.record:
            record_gif(args)
        else:
            play_with_render(args)
    except KeyboardInterrupt:
        print("\nDemo interrupted")


if __name__ == '__main__':
    main()
