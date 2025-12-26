"""
A3C Test Worker - 完全按照 ikostrikov/pytorch-a3c 实现
https://github.com/ikostrikov/pytorch-a3c
"""
import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic


def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

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
            recent_rewards.append(reward_sum)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            if avg_reward > best_avg:
                best_avg = avg_reward
                
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {:.1f}, avg(10) {:.2f}, best_avg {:.2f}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, avg_reward, best_avg, episode_length))
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(10)  # 减少等待时间便于观察

        state = torch.from_numpy(state)
