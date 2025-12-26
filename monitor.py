"""
A3C训练监控和可视化工具
用于记录训练过程中的各种指标，并生成可视化图表
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from collections import deque
from datetime import datetime

class TrainingMonitor:
    """训练监控器：记录和可视化训练过程"""
    
    def __init__(self, save_dir='./training_logs'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 记录各种指标
        self.episode_rewards = []  # 每个episode的奖励
        self.episode_lengths = []  # 每个episode的长度
        self.policy_losses = []    # 策略损失
        self.value_losses = []     # 价值损失
        self.entropy_losses = []   # 熵损失
        self.total_losses = []     # 总损失
        self.learning_rates = []   # 学习率变化
        self.timestamps = []       # 时间戳
        self.updates = []          # 更新次数
        
        # 滑动窗口用于计算平均值
        self.reward_window = deque(maxlen=100)
        self.avg_rewards = []      # 平均奖励（最近100个episode）
        
        self.start_time = time.time()
        self.total_steps = 0
        
    def log_episode(self, episode_reward, episode_length, process_id=0):
        """记录一个episode的信息"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.reward_window.append(episode_reward)
        self.avg_rewards.append(np.mean(self.reward_window))
        self.timestamps.append(time.time() - self.start_time)
        
    def log_losses(self, policy_loss, value_loss, entropy_loss, total_loss, update_count):
        """记录损失值"""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)
        self.total_losses.append(total_loss)
        self.updates.append(update_count)
        
    def log_learning_rate(self, lr):
        """记录学习率"""
        self.learning_rates.append(lr)
        
    def get_stats(self):
        """获取当前统计信息"""
        if len(self.episode_rewards) == 0:
            return {}
            
        return {
            'episodes': len(self.episode_rewards),
            'latest_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'avg_reward_100': np.mean(list(self.reward_window)) if self.reward_window else 0,
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'min_reward': min(self.episode_rewards) if self.episode_rewards else 0,
            'training_time': time.time() - self.start_time,
            'total_updates': len(self.policy_losses)
        }
        
    def plot_training_curves(self, save_path=None):
        """生成训练曲线图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('A3C Training Progress', fontsize=16, fontweight='bold')
        
        # 1. Episode奖励曲线
        ax1 = axes[0, 0]
        if self.episode_rewards:
            ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
            if len(self.avg_rewards) > 0:
                ax1.plot(self.avg_rewards, color='red', linewidth=2, label='Avg Reward (100 eps)')
            ax1.axhline(y=18, color='green', linestyle='--', label='Target (18)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 策略损失
        ax2 = axes[0, 1]
        if self.policy_losses:
            ax2.plot(self.policy_losses, color='purple', alpha=0.7)
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.set_title('Policy Loss')
            ax2.grid(True, alpha=0.3)
        
        # 3. 价值损失
        ax3 = axes[0, 2]
        if self.value_losses:
            ax3.plot(self.value_losses, color='orange', alpha=0.7)
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.set_title('Value Loss')
            ax3.grid(True, alpha=0.3)
        
        # 4. 熵损失
        ax4 = axes[1, 0]
        if self.entropy_losses:
            ax4.plot(self.entropy_losses, color='green', alpha=0.7)
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Entropy')
            ax4.set_title('Entropy Loss')
            ax4.grid(True, alpha=0.3)
        
        # 5. Episode长度
        ax5 = axes[1, 1]
        if self.episode_lengths:
            ax5.plot(self.episode_lengths, alpha=0.5, color='teal')
            # 添加滑动平均
            if len(self.episode_lengths) > 10:
                window = min(100, len(self.episode_lengths))
                avg_lengths = np.convolve(self.episode_lengths, 
                                          np.ones(window)/window, mode='valid')
                ax5.plot(range(window-1, len(self.episode_lengths)), 
                        avg_lengths, color='red', linewidth=2, label=f'Avg ({window} eps)')
                ax5.legend()
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Length')
            ax5.set_title('Episode Length')
            ax5.grid(True, alpha=0.3)
        
        # 6. 训练时间 vs 奖励
        ax6 = axes[1, 2]
        if self.timestamps and self.episode_rewards:
            ax6.plot(np.array(self.timestamps)/60, self.episode_rewards, 
                    alpha=0.3, color='blue')
            if len(self.avg_rewards) > 0:
                ax6.plot(np.array(self.timestamps[:len(self.avg_rewards)])/60, 
                        self.avg_rewards, color='red', linewidth=2)
            ax6.axhline(y=18, color='green', linestyle='--', label='Target')
            ax6.set_xlabel('Time (minutes)')
            ax6.set_ylabel('Reward')
            ax6.set_title('Reward vs Training Time')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Training curves saved to {save_path}')
        
    def plot_comparison(self, other_monitor, labels=['Original', 'Optimized'], save_path=None):
        """对比两次训练的结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('A3C Performance Comparison', fontsize=14, fontweight='bold')
        
        # 1. 奖励对比
        ax1 = axes[0]
        if self.avg_rewards:
            ax1.plot(self.avg_rewards, label=labels[0], linewidth=2)
        if other_monitor.avg_rewards:
            ax1.plot(other_monitor.avg_rewards, label=labels[1], linewidth=2)
        ax1.axhline(y=18, color='green', linestyle='--', alpha=0.5, label='Target')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward (100 eps)')
        ax1.set_title('Learning Curve Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收敛速度对比（时间）
        ax2 = axes[1]
        if self.timestamps and self.avg_rewards:
            ax2.plot(np.array(self.timestamps[:len(self.avg_rewards)])/60, 
                    self.avg_rewards, label=labels[0], linewidth=2)
        if other_monitor.timestamps and other_monitor.avg_rewards:
            ax2.plot(np.array(other_monitor.timestamps[:len(other_monitor.avg_rewards)])/60,
                    other_monitor.avg_rewards, label=labels[1], linewidth=2)
        ax2.axhline(y=18, color='green', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Sample Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 统计对比柱状图
        ax3 = axes[2]
        stats1 = self.get_stats()
        stats2 = other_monitor.get_stats()
        
        metrics = ['Max Reward', 'Final Avg', 'Training Time\n(min)']
        values1 = [
            stats1.get('max_reward', 0),
            stats1.get('avg_reward_100', 0),
            stats1.get('training_time', 0) / 60
        ]
        values2 = [
            stats2.get('max_reward', 0),
            stats2.get('avg_reward_100', 0),
            stats2.get('training_time', 0) / 60
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax3.bar(x - width/2, values1, width, label=labels[0], color='steelblue')
        ax3.bar(x + width/2, values2, width, label=labels[1], color='coral')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.set_title('Performance Metrics')
        ax3.legend()
        
        # 添加数值标签
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            ax3.text(i - width/2, v1 + 0.5, f'{v1:.1f}', ha='center', fontsize=9)
            ax3.text(i + width/2, v2 + 0.5, f'{v2:.1f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Comparison plot saved to {save_path}')
        
    def save_logs(self, filename=None):
        """保存训练日志到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'training_log_{timestamp}.json'
            
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'avg_rewards': self.avg_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'total_losses': self.total_losses,
            'timestamps': self.timestamps,
            'stats': self.get_stats()
        }
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f'Training logs saved to {filepath}')
        
    def load_logs(self, filepath):
        """从JSON文件加载训练日志"""
        with open(filepath, 'r') as f:
            log_data = json.load(f)
            
        self.episode_rewards = log_data.get('episode_rewards', [])
        self.episode_lengths = log_data.get('episode_lengths', [])
        self.avg_rewards = log_data.get('avg_rewards', [])
        self.policy_losses = log_data.get('policy_losses', [])
        self.value_losses = log_data.get('value_losses', [])
        self.entropy_losses = log_data.get('entropy_losses', [])
        self.total_losses = log_data.get('total_losses', [])
        self.timestamps = log_data.get('timestamps', [])
        print(f'Loaded training logs from {filepath}')


def print_training_summary(monitor, title="Training Summary"):
    """打印训练总结"""
    stats = monitor.get_stats()
    
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)
    print(f" Total Episodes:        {stats.get('episodes', 0)}")
    print(f" Total Updates:         {stats.get('total_updates', 0)}")
    print(f" Training Time:         {stats.get('training_time', 0)/60:.2f} minutes")
    print(f" Latest Reward:         {stats.get('latest_reward', 0):.1f}")
    print(f" Average Reward (100):  {stats.get('avg_reward_100', 0):.2f}")
    print(f" Max Reward:            {stats.get('max_reward', 0):.1f}")
    print(f" Min Reward:            {stats.get('min_reward', 0):.1f}")
    print("="*60 + "\n")


# 全局监控器实例（用于多进程共享）
_global_monitor = None

def get_global_monitor(save_dir='./training_logs'):
    """获取全局监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = TrainingMonitor(save_dir)
    return _global_monitor
