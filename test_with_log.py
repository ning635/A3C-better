"""
A3C Test Worker with Logging - 带日志记录的测试工作进程
用于可视化训练过程和结果分析
"""
import time
import os
import json
from collections import deque
from datetime import datetime

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic


def test_with_log(rank, args, shared_model, counter, log_dir="logs"):
    """
    带日志记录的测试函数
    
    Args:
        rank: 进程编号
        args: 参数配置
        shared_model: 共享模型
        counter: 全局步数计数器
        log_dir: 日志保存目录
    """
    torch.manual_seed(args.seed + rank)

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成唯一的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")
    
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    
    # 用于计算滚动平均
    recent_rewards = deque(maxlen=10)
    best_avg = -21.0
    
    # 日志数据
    episode_count = 0
    log_data = {
        "env_name": args.env_name,
        "start_time": timestamp,
        "hyperparameters": {
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "entropy_coef": args.entropy_coef,
            "value_loss_coef": args.value_loss_coef,
            "max_grad_norm": args.max_grad_norm,
            "num_processes": args.num_processes,
            "num_steps": args.num_steps
        },
        "episodes": []
    }

    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].item()

        state, reward, done, _ = env.step(action)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward
        actions.append(action)

        # a quick hack to prevent the agent from stucking
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            episode_count += 1
            recent_rewards.append(reward_sum)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            elapsed_time = time.time() - start_time
            
            if avg_reward > best_avg:
                best_avg = avg_reward
            
            # 记录日志数据
            episode_data = {
                "episode": episode_count,
                "elapsed_time": elapsed_time,
                "total_steps": counter.value,
                "fps": counter.value / elapsed_time if elapsed_time > 0 else 0,
                "episode_reward": reward_sum,
                "avg_reward_10": avg_reward,
                "best_avg": best_avg,
                "episode_length": episode_length
            }
            log_data["episodes"].append(episode_data)
            
            # 定期保存日志（每10个episode或每次打破记录时）
            if episode_count % 10 == 0 or avg_reward == best_avg:
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {:.1f}, avg(10) {:.2f}, best_avg {:.2f}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
                counter.value, counter.value / elapsed_time,
                reward_sum, avg_reward, best_avg, episode_length))
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(10)  # 减少等待时间便于观察

        state = torch.from_numpy(state)
