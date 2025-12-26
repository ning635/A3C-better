"""
PPO Actor-Critic Model for Atari
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """PPO Actor-Critic 网络 (Nature CNN)"""
    
    def __init__(self, num_actions):
        super().__init__()
        
        # CNN 特征提取器 (Nature DQN 架构)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        
        # Actor (策略网络)
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        
        # Critic (价值网络)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)
    
    def get_value(self, x):
        """获取状态价值"""
        return self.critic(self.network(x / 255.0))
    
    def get_action_and_value(self, x, action=None):
        """获取动作和价值"""
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
