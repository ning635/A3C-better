"""
A3C 对比实验脚本
用于对比原始A3C和优化版A3C的性能

使用方法：
    python run_comparison.py
    
或者分别运行：
    python run_comparison.py --mode original   # 只运行原始版本
    python run_comparison.py --mode optimized  # 只运行优化版本
    python run_comparison.py --mode plot       # 只生成对比图表
"""

import argparse
import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time


def run_experiment(mode, n_workers=4, max_episodes=500):
    """运行实验"""
    print(f"\n{'='*60}")
    print(f" Running {mode.upper()} A3C")
    print(f"{'='*60}\n")
    
    if mode == 'original':
        cmd = f"python main.py --n-workers {n_workers} --env-name PongNoFrameskip-v4"
    else:
        cmd = f"python main_optimized.py --n-workers {n_workers} --env-name PongNoFrameskip-v4 --use-gae True"
    
    start_time = time.time()
    subprocess.run(cmd, shell=True)
    elapsed_time = time.time() - start_time
    
    return elapsed_time


def load_scores(filename):
    """从文件加载分数数据"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def calculate_metrics(scores_avg):
    """计算性能指标"""
    if not scores_avg or len(scores_avg) < 100:
        return {}
    
    scores = np.array(scores_avg)
    
    # 找到首次达到目标的episode
    target = 18.0
    episodes_to_target = None
    for i, score in enumerate(scores):
        if i >= 99:  # 需要至少100个episode才能计算平均
            avg_100 = np.mean(scores[max(0, i-99):i+1])
            if avg_100 >= target:
                episodes_to_target = i + 1
                break
    
    return {
        'total_episodes': len(scores),
        'max_score': float(np.max(scores)),
        'final_avg_100': float(np.mean(scores[-100:])) if len(scores) >= 100 else float(np.mean(scores)),
        'episodes_to_target': episodes_to_target,
        'std_score': float(np.std(scores)),
    }


def plot_comparison(original_scores, optimized_scores, save_path='comparison_results.png'):
    """生成对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('A3C vs A3C-GAE Performance Comparison', fontsize=16, fontweight='bold')
    
    # 颜色方案
    color_original = '#2E86AB'  # 蓝色
    color_optimized = '#E94F37'  # 红色
    
    # 1. 学习曲线对比
    ax1 = axes[0, 0]
    if original_scores:
        ax1.plot(original_scores, alpha=0.3, color=color_original)
        # 滑动平均
        window = min(50, len(original_scores))
        avg_original = np.convolve(original_scores, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(original_scores)), avg_original, 
                color=color_original, linewidth=2, label='Original A3C')
    
    if optimized_scores:
        ax1.plot(optimized_scores, alpha=0.3, color=color_optimized)
        window = min(50, len(optimized_scores))
        avg_optimized = np.convolve(optimized_scores, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(optimized_scores)), avg_optimized,
                color=color_optimized, linewidth=2, label='A3C + GAE')
    
    ax1.axhline(y=18, color='green', linestyle='--', alpha=0.7, label='Target (18)')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Learning Curves', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 分数分布对比（箱线图）
    ax2 = axes[0, 1]
    data = []
    labels = []
    if original_scores:
        data.append(original_scores)
        labels.append('Original')
    if optimized_scores:
        data.append(optimized_scores)
        labels.append('GAE Optimized')
    
    if data:
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        colors = [color_original, color_optimized][:len(data)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.axhline(y=18, color='green', linestyle='--', alpha=0.7, label='Target')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Score Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 收敛速度对比
    ax3 = axes[1, 0]
    
    # 计算滑动平均
    def running_avg(scores, window=100):
        if len(scores) < window:
            return scores
        return [np.mean(scores[max(0, i-window+1):i+1]) for i in range(len(scores))]
    
    if original_scores and len(original_scores) > 0:
        avg_orig = running_avg(original_scores)
        ax3.plot(avg_orig, color=color_original, linewidth=2, label='Original A3C')
    
    if optimized_scores and len(optimized_scores) > 0:
        avg_opt = running_avg(optimized_scores)
        ax3.plot(avg_opt, color=color_optimized, linewidth=2, label='A3C + GAE')
    
    ax3.axhline(y=18, color='green', linestyle='--', alpha=0.7, label='Target')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Average Score (100 eps)', fontsize=12)
    ax3.set_title('Convergence Comparison', fontsize=14)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能指标对比（柱状图）
    ax4 = axes[1, 1]
    
    metrics_orig = calculate_metrics(original_scores) if original_scores else {}
    metrics_opt = calculate_metrics(optimized_scores) if optimized_scores else {}
    
    metric_names = ['Max Score', 'Final Avg', 'Std Dev']
    orig_values = [
        metrics_orig.get('max_score', 0),
        metrics_orig.get('final_avg_100', 0),
        metrics_orig.get('std_score', 0)
    ]
    opt_values = [
        metrics_opt.get('max_score', 0),
        metrics_opt.get('final_avg_100', 0),
        metrics_opt.get('std_score', 0)
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, orig_values, width, label='Original', color=color_original, alpha=0.8)
    bars2 = ax4.bar(x + width/2, opt_values, width, label='GAE Optimized', color=color_optimized, alpha=0.8)
    
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Performance Metrics', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, orig_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, opt_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved to: {save_path}")


def print_summary(original_metrics, optimized_metrics):
    """打印性能对比总结"""
    print("\n" + "="*70)
    print(" PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    headers = ['Metric', 'Original A3C', 'A3C + GAE', 'Improvement']
    print(f"\n{headers[0]:<25} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15}")
    print("-"*70)
    
    metrics = [
        ('Max Score', 'max_score', 'higher'),
        ('Final Avg (100 eps)', 'final_avg_100', 'higher'),
        ('Score Std Dev', 'std_score', 'lower'),
        ('Episodes to Target', 'episodes_to_target', 'lower'),
    ]
    
    for name, key, better in metrics:
        orig_val = original_metrics.get(key, 'N/A')
        opt_val = optimized_metrics.get(key, 'N/A')
        
        if isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)):
            if better == 'higher':
                improvement = ((opt_val - orig_val) / abs(orig_val) * 100) if orig_val != 0 else 0
            else:
                improvement = ((orig_val - opt_val) / abs(orig_val) * 100) if orig_val != 0 else 0
            
            orig_str = f"{orig_val:.2f}" if isinstance(orig_val, float) else str(orig_val)
            opt_str = f"{opt_val:.2f}" if isinstance(opt_val, float) else str(opt_val)
            imp_str = f"{improvement:+.1f}%" if improvement != 0 else "N/A"
        else:
            orig_str = str(orig_val) if orig_val else 'N/A'
            opt_str = str(opt_val) if opt_val else 'N/A'
            imp_str = 'N/A'
        
        print(f"{name:<25} {orig_str:<15} {opt_str:<15} {imp_str:<15}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='A3C Comparison Experiment')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'original', 'optimized', 'plot'],
                        help='Which experiment to run')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--original-log', type=str, default='training_logs/original_scores.json',
                        help='Path to original training log')
    parser.add_argument('--optimized-log', type=str, default='training_logs/optimized_scores.json',
                        help='Path to optimized training log')
    
    args = parser.parse_args()
    
    os.makedirs('training_logs', exist_ok=True)
    
    print("\n" + "="*60)
    print(" A3C COMPARISON EXPERIMENT")
    print(f" Mode: {args.mode}")
    print(f" Workers: {args.n_workers}")
    print("="*60)
    
    original_scores = None
    optimized_scores = None
    
    if args.mode in ['all', 'original']:
        print("\n[1/2] Training Original A3C...")
        run_experiment('original', args.n_workers)
        # 这里需要从训练中保存分数
        
    if args.mode in ['all', 'optimized']:
        print("\n[2/2] Training Optimized A3C...")
        run_experiment('optimized', args.n_workers)
    
    if args.mode == 'plot':
        # 加载之前保存的数据
        if os.path.exists(args.original_log):
            with open(args.original_log, 'r') as f:
                original_scores = json.load(f)
        if os.path.exists(args.optimized_log):
            with open(args.optimized_log, 'r') as f:
                optimized_scores = json.load(f)
        
        if original_scores or optimized_scores:
            plot_comparison(original_scores, optimized_scores)
            
            orig_metrics = calculate_metrics(original_scores) if original_scores else {}
            opt_metrics = calculate_metrics(optimized_scores) if optimized_scores else {}
            print_summary(orig_metrics, opt_metrics)
    
    print("\n✓ Experiment complete!")


if __name__ == '__main__':
    main()
