"""
A3C 游戏演示脚本 (简化版)
直接使用训练环境进行演示，支持实时渲染和GIF录制

使用方法:
    python demo_game.py                         # 无模型演示（随机动作）
    python demo_game.py --model saved.pt        # 使用训练好的模型
    python demo_game.py --record --episodes 3   # 录制GIF
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

# 检查可用的显示库
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("提示: 安装opencv可获得实时显示: pip install opencv-python")

IMAGEIO_AVAILABLE = False
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    print("提示: 安装imageio可录制GIF: pip install imageio")


def load_model(model_path, num_inputs, action_space):
    """加载模型"""
    from model import ActorCritic
    model = ActorCritic(num_inputs, action_space)
    
    if model_path and os.path.exists(model_path):
        print(f"[OK] Loading model: {model_path}")
        try:
            # 尝试加载state_dict
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' not in str(type(state_dict)):
                model.load_state_dict(state_dict)
            else:
                # 可能是完整模型
                model = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Load failed: {e}")
            print("Using random initialized model")
    else:
        print("[WARN] Model not found, using random initialization (agent will act randomly)")
    
    model.eval()
    return model


def demo_with_cv2(env, model, args):
    """使用OpenCV进行实时演示"""
    print("\n" + "="*50)
    print(" A3C Game Demo - Real-time Rendering")
    print("="*50)
    print(f" Environment: {args.env_name}")
    print(f" Episodes: {args.episodes}")
    print(" Press 'q' to quit, 's' to screenshot")
    print("="*50 + "\n")
    
    episode_rewards = []
    
    for ep in range(args.episodes):
        state = env.reset()
        state_tensor = torch.from_numpy(state)
        
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done:
            step += 1
            
            # 选择动作
            with torch.no_grad():
                value, logit, (hx, cx) = model((state_tensor.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1)[1].item()
            
            # 执行动作
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            state_tensor = torch.from_numpy(state)
            
            # 渲染
            frame = state[0] if len(state.shape) == 3 else state
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, (420, 420), interpolation=cv2.INTER_NEAREST)
            
            # 添加信息
            info_bar = np.zeros((60, 420), dtype=np.uint8)
            cv2.putText(info_bar, f"Episode: {ep+1}/{args.episodes}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            cv2.putText(info_bar, f"Reward: {episode_reward:.0f}  Step: {step}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            
            display = np.vstack([info_bar, frame])
            cv2.imshow('A3C Pong Demo', display)
            
            key = cv2.waitKey(args.delay) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return episode_rewards
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_ep{ep+1}_step{step}.png', display)
                print(f"Screenshot saved: screenshot_ep{ep+1}_step{step}.png")
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward = {episode_reward:.0f}")
    
    cv2.destroyAllWindows()
    return episode_rewards


def demo_record_gif(env, model, args):
    """录制GIF"""
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio required: pip install imageio")
        return []
    
    print("\n" + "="*50)
    print(" A3C Game Demo - Recording GIF")
    print("="*50)
    print(f" Environment: {args.env_name}")
    print(f" Episodes: {args.episodes}")
    print("="*50 + "\n")
    
    os.makedirs('videos', exist_ok=True)
    episode_rewards = []
    
    for ep in range(args.episodes):
        frames = []
        state = env.reset()
        state_tensor = torch.from_numpy(state)
        
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                value, logit, (hx, cx) = model((state_tensor.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1)[1].item()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            state_tensor = torch.from_numpy(state)
            
            # 保存帧
            frame = state[0] if len(state.shape) == 3 else state
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, (168, 168), interpolation=cv2.INTER_NEAREST) if CV2_AVAILABLE else frame
            frames.append(frame)
        
        episode_rewards.append(episode_reward)
        
        # 保存GIF
        gif_path = f'videos/episode_{ep+1}_reward_{int(episode_reward)}.gif'
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Episode {ep+1}: Reward = {episode_reward:.0f} -> {gif_path}")
    
    return episode_rewards


def demo_text_only(env, model, args):
    """纯文本模式演示（无需GUI）"""
    print("\n" + "="*50)
    print(" A3C Game Demo - Text Mode")
    print("="*50)
    print(f" Environment: {args.env_name}")
    print(f" Episodes: {args.episodes}")
    print("="*50 + "\n")
    
    episode_rewards = []
    
    for ep in range(args.episodes):
        state = env.reset()
        state_tensor = torch.from_numpy(state)
        
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done:
            step += 1
            
            with torch.no_grad():
                value, logit, (hx, cx) = model((state_tensor.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1)[1].item()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            state_tensor = torch.from_numpy(state)
            
            # 每100步打印一次
            if step % 100 == 0:
                print(f"  Step {step}, Current Reward: {episode_reward:.0f}")
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Final Reward = {episode_reward:.0f}, Steps = {step}")
    
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description='A3C 游戏演示')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--episodes', type=int, default=3, help='演示episode数')
    parser.add_argument('--delay', type=int, default=1, help='帧间延迟(ms)')
    parser.add_argument('--record', action='store_true', help='录制GIF')
    parser.add_argument('--text', action='store_true', help='纯文本模式')
    
    args = parser.parse_args()
    
    # 创建环境
    from envs import create_atari_env
    env = create_atari_env(args.env_name)
    
    # 加载模型
    model = load_model(args.model, env.observation_space.shape[0], env.action_space)
    
    # 运行演示
    try:
        if args.record:
            rewards = demo_record_gif(env, model, args)
        elif args.text or not CV2_AVAILABLE:
            rewards = demo_text_only(env, model, args)
        else:
            rewards = demo_with_cv2(env, model, args)
    except KeyboardInterrupt:
        print("\nDemo interrupted")
        rewards = []
    finally:
        env.close()
    
    # 打印统计
    if rewards:
        print("\n" + "="*50)
        print(" Demo Statistics")
        print("="*50)
        print(f" Episodes: {len(rewards)}")
        print(f" Average Reward: {np.mean(rewards):.1f}")
        print(f" Max Reward: {max(rewards):.0f}")
        print(f" Min Reward: {min(rewards):.0f}")
        print("="*50)


if __name__ == '__main__':
    main()
