"""
对比可视化脚本 - 自动运行基线和优化版本并生成对比图
"""
import subprocess
import time
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np


def run_training(script_name, log_file, duration_minutes=30, description=""):
    """
    运行训练脚本并等待指定时间
    
    Args:
        script_name: 训练脚本名称
        log_file: 日志文件路径
        duration_minutes: 运行时间（分钟）
        description: 描述信息
    """
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f" 脚本: {script_name}")
    print(f" 日志: {log_file}")
    print(f" 运行时间: {duration_minutes} 分钟")
    print(f"{'='*60}\n")
    
    # 启动训练进程
    process = subprocess.Popen([sys.executable, script_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
    
    # 等待指定时间
    time.sleep(duration_minutes * 60)
    
    # 终止进程
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    
    print(f"{description} 训练完成！")
    return log_file


def generate_comparison_plot(baseline_log, optimized_log, output_dir="figures"):
    """
    生成基线 vs 优化版本的对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载日志
    with open(baseline_log, 'r') as f:
        baseline_data = json.load(f)
    with open(optimized_log, 'r') as f:
        optimized_data = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Episode奖励对比
    ax1 = axes[0, 0]
    baseline_eps = baseline_data["episodes"]
    optimized_eps = optimized_data["episodes"]
    
    ax1.plot([ep["episode"] for ep in baseline_eps], 
             [ep["avg_reward_10"] for ep in baseline_eps], 
             'b-', linewidth=2, label='Baseline A3C')
    ax1.plot([ep["episode"] for ep in optimized_eps], 
             [ep["avg_reward_10"] for ep in optimized_eps], 
             'r-', linewidth=2, label='Optimized A3C')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (10 eps)')
    ax1.set_title('按Episode的平均奖励对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 按时间对比
    ax2 = axes[0, 1]
    ax2.plot([ep["elapsed_time"]/60 for ep in baseline_eps], 
             [ep["avg_reward_10"] for ep in baseline_eps], 
             'b-', linewidth=2, label='Baseline A3C')
    ax2.plot([ep["elapsed_time"]/60 for ep in optimized_eps], 
             [ep["avg_reward_10"] for ep in optimized_eps], 
             'r-', linewidth=2, label='Optimized A3C')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Average Reward (10 eps)')
    ax2.set_title('按时间的平均奖励对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 按步数对比
    ax3 = axes[1, 0]
    ax3.plot([ep["total_steps"]/1e6 for ep in baseline_eps], 
             [ep["avg_reward_10"] for ep in baseline_eps], 
             'b-', linewidth=2, label='Baseline A3C')
    ax3.plot([ep["total_steps"]/1e6 for ep in optimized_eps], 
             [ep["avg_reward_10"] for ep in optimized_eps], 
             'r-', linewidth=2, label='Optimized A3C')
    ax3.set_xlabel('Total Steps (M)')
    ax3.set_ylabel('Average Reward (10 eps)')
    ax3.set_title('按步数的平均奖励对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终性能对比
    ax4 = axes[1, 1]
    
    # 计算最后10个episode的平均
    baseline_final = np.mean([ep["episode_reward"] for ep in baseline_eps[-10:]])
    optimized_final = np.mean([ep["episode_reward"] for ep in optimized_eps[-10:]])
    
    bars = ax4.bar(['Baseline A3C', 'Optimized A3C'], 
                   [baseline_final, optimized_final],
                   color=['steelblue', 'coral'], alpha=0.7)
    
    # 添加数值标签
    for bar, val in zip(bars, [baseline_final, optimized_final]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4.set_ylabel('Final Average Reward')
    ax4.set_title('最终性能对比')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 计算提升百分比
    improvement = (optimized_final - baseline_final) / abs(baseline_final) * 100 if baseline_final != 0 else 0
    ax4.text(0.5, 0.9, f'提升: {improvement:.1f}%', 
             transform=ax4.transAxes, fontsize=14, fontweight='bold',
             ha='center', color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'baseline_vs_optimized.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图已保存到: {output_file}")
    return output_file


def print_comparison_stats(baseline_log, optimized_log):
    """打印对比统计信息"""
    with open(baseline_log, 'r') as f:
        baseline_data = json.load(f)
    with open(optimized_log, 'r') as f:
        optimized_data = json.load(f)
    
    baseline_eps = baseline_data["episodes"]
    optimized_eps = optimized_data["episodes"]
    
    print("\n" + "="*60)
    print(" 训练对比统计")
    print("="*60)
    
    # 基线统计
    baseline_rewards = [ep["episode_reward"] for ep in baseline_eps]
    baseline_final = np.mean(baseline_rewards[-10:]) if len(baseline_rewards) >= 10 else np.mean(baseline_rewards)
    
    print("\n[Baseline A3C]")
    print(f"  总Episode数: {len(baseline_eps)}")
    print(f"  训练时间: {baseline_eps[-1]['elapsed_time']/60:.1f} 分钟")
    print(f"  最高奖励: {max(baseline_rewards):.1f}")
    print(f"  最终平均: {baseline_final:.1f}")
    print(f"  最佳平均: {baseline_eps[-1]['best_avg']:.1f}")
    
    # 优化版统计
    optimized_rewards = [ep["episode_reward"] for ep in optimized_eps]
    optimized_final = np.mean(optimized_rewards[-10:]) if len(optimized_rewards) >= 10 else np.mean(optimized_rewards)
    
    print("\n[Optimized A3C]")
    print(f"  总Episode数: {len(optimized_eps)}")
    print(f"  训练时间: {optimized_eps[-1]['elapsed_time']/60:.1f} 分钟")
    print(f"  最高奖励: {max(optimized_rewards):.1f}")
    print(f"  最终平均: {optimized_final:.1f}")
    print(f"  最佳平均: {optimized_eps[-1]['best_avg']:.1f}")
    
    # 对比
    print("\n[对比结果]")
    improvement = optimized_final - baseline_final
    improvement_pct = (improvement / abs(baseline_final) * 100) if baseline_final != 0 else 0
    print(f"  平均奖励提升: {improvement:.1f} ({improvement_pct:+.1f}%)")
    
    best_improvement = optimized_eps[-1]['best_avg'] - baseline_eps[-1]['best_avg']
    print(f"  最佳平均提升: {best_improvement:.1f}")
    
    print("="*60 + "\n")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='A3C 基线 vs 优化版本对比')
    parser.add_argument('--baseline-log', type=str, default=None,
                        help='基线训练日志文件')
    parser.add_argument('--optimized-log', type=str, default=None,
                        help='优化版训练日志文件')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='输出目录')
    
    args = parser.parse_args()
    
    if args.baseline_log and args.optimized_log:
        # 使用提供的日志文件
        generate_comparison_plot(args.baseline_log, args.optimized_log, args.output_dir)
        print_comparison_stats(args.baseline_log, args.optimized_log)
    else:
        print("请提供基线和优化版本的日志文件路径")
        print("用法: python compare_versions.py --baseline-log logs/baseline.json --optimized-log logs/optimized.json")


if __name__ == '__main__':
    main()
