"""
Atari Environment Wrappers for PPO
Based on stable-baselines3 and CleanRL implementations
"""
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    FrameStackObservation,
)
import cv2


class NoopResetEnv(gym.Wrapper):
    """在reset时执行随机数量的NOOP动作"""
    
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """每个动作重复skip帧，返回最后2帧的最大值"""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    
    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """将生命损失视为episode结束（仅用于训练）"""
    
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """在reset后执行FIRE动作（部分游戏需要）"""
    
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT (任意非FIRE动作)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """将奖励裁剪到 {-1, 0, +1}"""
    
    def reward(self, reward):
        return np.sign(reward)


class ResizeObservation(gym.ObservationWrapper):
    """调整观察图像大小"""
    
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        obs_shape = self.shape + (1,) if len(env.observation_space.shape) == 2 else self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        if len(observation.shape) == 2:
            observation = observation[:, :, np.newaxis]
        return observation


class GrayScaleObservation(gym.ObservationWrapper):
    """转换为灰度图"""
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape[:2] + (1,)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation[:, :, np.newaxis]


def make_env(env_id, seed, idx, capture_video=False, run_name=""):
    """创建单个环境的工厂函数"""
    
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        env = RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        env = ClipRewardEnv(env)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env)
        env = FrameStackObservation(env, 4)
        
        env.action_space.seed(seed)
        return env
    
    return thunk
