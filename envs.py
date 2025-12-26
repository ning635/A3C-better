"""
Environment wrappers - 适配 gymnasium
https://github.com/ikostrikov/pytorch-a3c
添加帧堆叠以提供时序信息
"""
import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from collections import deque


def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = FrameStack(env, 4)  # 添加4帧堆叠
    env = NormalizedEnv(env)
    env = GymnasiumCompatWrapper(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping)
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42], dtype=np.float32)

    def observation(self, observation):
        return _process_frame42(observation)


class FrameStack(gym.Wrapper):
    """堆叠最近k帧作为观察"""
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(shp[0] * k, shp[1], shp[2]),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        # 继承上游env的observation_space（已经是4通道了）
        self.observation_space = env.observation_space
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class GymnasiumCompatWrapper(gym.Wrapper):
    """Wrapper to make gymnasium compatible with old gym API"""
    def __init__(self, env):
        super().__init__(env)
        self._seed = None
    
    def seed(self, seed=None):
        self._seed = seed
        
    def reset(self, **kwargs):
        if self._seed is not None:
            kwargs['seed'] = self._seed
            self._seed = None
        obs, info = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
