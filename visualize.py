"""
A3C 训练结果可视化脚本
生成训练曲线图、直方图等可视化图表

使用方法:
python visualize.py logs/training_log_*.json

或者指定多个日志文件进行对比:
python visualize.py logs/baseline.json logs/optimized.json --compare
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np


def load_log(log_file):
    """加载训练日志文件"""
    with open(log_file, 'r') as f:
        return json.load(f)


def plot_training_curve(log_data, output_dir="figures", title_prefix=""):
    """
    绘制训练曲线图
    
    Args:
        log_data: 日志数据
        output_dir: 输出目录
        title_prefix: 标题前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    
    episodes = log_data["episodes"]
    episode_nums = [ep["episode"] for ep in episodes]
    rewards = [ep["episode_reward"] for ep in episodes]
    avg_rewards = [ep["avg_reward_10"] for ep in episodes]
    best_avgs = [ep["best_avg"] for ep in episodes]
    
    # 1. 训练奖励曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_nums, rewards, 'b-', alpha=0.3, label='Episode Reward')
    plt.plot(episode_nums, avg_rewards, 'r-', linewidth=2, label='Avg Reward (10 eps)')
    plt.plot(episode_nums, best_avgs, 'g--', linewidth=2, label='Best Avg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{title_prefix}训练奖励曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 按时间的奖励曲线
    plt.subplot(2, 2, 2)
    times = [ep["elapsed_time"] / 60 for ep in episodes]  # 转换为分钟
    plt.plot(times, rewards, 'b-', alpha=0.3, label='Episode Reward')
    plt.plot(times, avg_rewards, 'r-', linewidth=2, label='Avg Reward (10 eps)')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Reward')
    plt.title(f'{title_prefix}时间-奖励曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 按总步数的奖励曲线
    plt.subplot(2, 2, 3)
    steps = [ep["total_steps"] / 1e6 for ep in episodes]  # 转换为百万步
    plt.plot(steps, rewards, 'b-', alpha=0.3, label='Episode Reward')
    plt.plot(steps, avg_rewards, 'r-', linewidth=2, label='Avg Reward (10 eps)')
    plt.xlabel('Total Steps (M)')
    plt.ylabel('Reward')
    plt.title(f'{title_prefix}步数-奖励曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. FPS曲线
    plt.subplot(2, 2, 4)
    fps = [ep["fps"] for ep in episodes]
    plt.plot(episode_nums, fps, 'purple', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('FPS')
    plt.title(f'{title_prefix}训练速度 (FPS)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{title_prefix}training_curves.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {output_file}")
    
    return output_file


def plot_reward_histogram(log_data, output_dir="figures", title_prefix=""):
    """
    绘制奖励分布直方图
    
    Args:
        log_data: 日志数据
        output_dir: 输出目录
        title_prefix: 标题前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    
    episodes = log_data["episodes"]
    rewards = [ep["episode_reward"] for ep in episodes]
    
    plt.figure(figsize=(12, 5))
    
    # 1. 整体奖励分布
    plt.subplot(1, 2, 1)
    plt.hist(rewards, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix}奖励分布直方图')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    plt.axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 训练前后对比 (前25% vs 后25%)
    plt.subplot(1, 2, 2)
    n = len(rewards)
    quarter = n // 4
    early_rewards = rewards[:quarter] if quarter > 0 else rewards[:n//2]
    late_rewards = rewards[-quarter:] if quarter > 0 else rewards[n//2:]
    
    plt.hist(early_rewards, bins=20, color='lightcoral', edgecolor='white', 
             alpha=0.7, label=f'Early (first {len(early_rewards)} eps)')
    plt.hist(late_rewards, bins=20, color='lightgreen', edgecolor='white', 
             alpha=0.7, label=f'Late (last {len(late_rewards)} eps)')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix}训练前后奖励对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{title_prefix}reward_histogram.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"直方图已保存到: {output_file}")
    
    return output_file


def plot_episode_length(log_data, output_dir="figures", title_prefix=""):
    """
    绘制episode长度变化图
    
    Args:
        log_data: 日志数据
        output_dir: 输出目录
        title_prefix: 标题前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    
    episodes = log_data["episodes"]
    episode_nums = [ep["episode"] for ep in episodes]
    lengths = [ep["episode_length"] for ep in episodes]
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_nums, lengths, 'steelblue', alpha=0.7)
    
    # 添加移动平均线
    window = min(10, len(lengths))
    if window > 1:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        plt.plot(episode_nums[window-1:], moving_avg, 'red', linewidth=2, label=f'{window}-episode moving avg')
        plt.legend()
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title(f'{title_prefix}Episode 长度变化')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{title_prefix}episode_length.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Episode长度图已保存到: {output_file}")
    
    return output_file


def plot_comparison(log_files, labels, output_dir="figures"):
    """
    对比多个训练日志
    
    Args:
        log_files: 日志文件列表
        labels: 标签列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(14, 10))
    
    # 1. 奖励曲线对比
    plt.subplot(2, 2, 1)
    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        log_data = load_log(log_file)
        episodes = log_data["episodes"]
        episode_nums = [ep["episode"] for ep in episodes]
        avg_rewards = [ep["avg_reward_10"] for ep in episodes]
        plt.plot(episode_nums, avg_rewards, color=colors[i % len(colors)], 
                linewidth=2, label=label)
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward (10 eps)')
    plt.title('平均奖励对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 按时间的对比
    plt.subplot(2, 2, 2)
    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        log_data = load_log(log_file)
        episodes = log_data["episodes"]
        times = [ep["elapsed_time"] / 60 for ep in episodes]
        avg_rewards = [ep["avg_reward_10"] for ep in episodes]
        plt.plot(times, avg_rewards, color=colors[i % len(colors)], 
                linewidth=2, label=label)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Avg Reward (10 eps)')
    plt.title('时间-奖励对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 按步数的对比
    plt.subplot(2, 2, 3)
    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        log_data = load_log(log_file)
        episodes = log_data["episodes"]
        steps = [ep["total_steps"] / 1e6 for ep in episodes]
        avg_rewards = [ep["avg_reward_10"] for ep in episodes]
        plt.plot(steps, avg_rewards, color=colors[i % len(colors)], 
                linewidth=2, label=label)
    plt.xlabel('Total Steps (M)')
    plt.ylabel('Avg Reward (10 eps)')
    plt.title('步数-奖励对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 最终性能对比 (柱状图)
    plt.subplot(2, 2, 4)
    final_scores = []
    for log_file in log_files:
        log_data = load_log(log_file)
        episodes = log_data["episodes"]
        # 取最后10个episode的平均
        last_n = min(10, len(episodes))
        final_avg = np.mean([ep["episode_reward"] for ep in episodes[-last_n:]])
        final_scores.append(final_avg)
    
    bars = plt.bar(labels, final_scores, color=colors[:len(labels)], alpha=0.7)
    plt.ylabel('Final Avg Reward')
    plt.title('最终性能对比')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, score in zip(bars, final_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{score:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存到: {output_file}")
    
    return output_file


def print_statistics(log_data):
    """打印训练统计信息"""
    episodes = log_data["episodes"]
    rewards = [ep["episode_reward"] for ep in episodes]
    
    print("\n" + "="*50)
    print("训练统计信息")
    print("="*50)
    print(f"环境: {log_data['env_name']}")
    print(f"总Episode数: {len(episodes)}")
    print(f"总训练时间: {episodes[-1]['elapsed_time']/60:.1f} 分钟")
    print(f"总步数: {episodes[-1]['total_steps']:,}")
    print("-"*50)
    print(f"最高单Episode奖励: {max(rewards):.1f}")
    print(f"最低单Episode奖励: {min(rewards):.1f}")
    print(f"平均奖励: {np.mean(rewards):.2f}")
    print(f"奖励标准差: {np.std(rewards):.2f}")
    print(f"最佳10Episode平均: {episodes[-1]['best_avg']:.2f}")
    print("-"*50)
    print("超参数配置:")
    for key, value in log_data["hyperparameters"].items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")


def visualize_all(log_file, output_dir="figures", title_prefix=""):
    """
    生成所有可视化图表
    
    Args:
        log_file: 日志文件路径
        output_dir: 输出目录
        title_prefix: 标题前缀
    """
    log_data = load_log(log_file)
    
    print(f"\n正在处理日志文件: {log_file}")
    print_statistics(log_data)
    
    files = []
    files.append(plot_training_curve(log_data, output_dir, title_prefix))
    files.append(plot_reward_histogram(log_data, output_dir, title_prefix))
    files.append(plot_episode_length(log_data, output_dir, title_prefix))
    
    print(f"\n所有图表已保存到 {output_dir}/ 目录")
    return files


def main():
    parser = argparse.ArgumentParser(description='A3C 训练结果可视化')
    parser.add_argument('log_files', nargs='+', help='训练日志文件(JSON格式)')
    parser.add_argument('--output-dir', '-o', type=str, default='figures',
                        help='输出目录 (default: figures)')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='对比多个日志文件')
    parser.add_argument('--labels', '-l', nargs='+', default=None,
                        help='对比时的标签名称')
    
    args = parser.parse_args()
    
    if args.compare and len(args.log_files) > 1:
        labels = args.labels if args.labels else [f'Model {i+1}' for i in range(len(args.log_files))]
        plot_comparison(args.log_files, labels, args.output_dir)
    else:
        for log_file in args.log_files:
            prefix = os.path.splitext(os.path.basename(log_file))[0] + "_"
            visualize_all(log_file, args.output_dir, prefix)


if __name__ == '__main__':
    main()
