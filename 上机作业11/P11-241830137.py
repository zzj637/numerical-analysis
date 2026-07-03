import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 蒲丰投针问题 ====================
def buffon_needle(num_trials, needle_length=1.0):
    """单次蒲丰投针实验，返回π的估计值"""
    line_spacing = 2.0 * needle_length  # d = 2h
    hits = 0
    
    for _ in range(num_trials):
        distance = random.uniform(0, needle_length)
        angle = random.uniform(0, math.pi / 2.0)
        projection = (needle_length / 2.0) * math.sin(angle)
        
        if projection >= distance:
            hits += 1
    
    if hits == 0:
        return float('inf')
    
    probability = hits / num_trials
    pi_estimate = 1.0 / probability
    return pi_estimate

# 进行大量蒲丰投针实验，估计π的值，并记录实验次数与误差的关系
def buffon_needle_experiments():
    print("=" * 60)
    print("1. 蒲丰投针问题模拟")
    print("=" * 60)
    
    trial_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
    pi_estimates = []
    errors = []
    
    print("\n实验次数\tπ估计值\t\t绝对误差")
    print("-" * 50)
    
    for n in trial_counts:
        pi_est = buffon_needle(n)
        error = abs(pi_est - math.pi)
        pi_estimates.append(pi_est)
        errors.append(error)
        print(f"{n}\t\t{pi_est:.6f}\t{error:.6f}")

# ==================== 2. 蒙特卡洛积分 ====================
def f(x):
    """被积函数 f(x) = e^(-x)"""
    return np.exp(-x)

def monte_carlo_hit_or_miss(n, a=0, b=1, y_max=1):
    """随机投点法计算定积分"""
    hits = 0
    hit_xs = []
    hit_ys = []
    miss_xs = []
    miss_ys = []
    
    for _ in range(n):
        x = random.uniform(a, b)
        y = random.uniform(0, y_max)
        
        if y <= f(x):
            hits += 1
            hit_xs.append(x)
            hit_ys.append(y)
        else:
            miss_xs.append(x)
            miss_ys.append(y)
    
    area_rect = (b - a) * y_max
    estimate = area_rect * hits / n
    return estimate, hit_xs, hit_ys, miss_xs, miss_ys

def monte_carlo_sample_mean(n, a=0, b=1):
    """样本平均法计算定积分"""
    samples = [random.uniform(a, b) for _ in range(n)]
    func_values = [f(x) for x in samples]
    estimate = (b - a) * sum(func_values) / n
    return estimate, samples, func_values

def integral_experiment():
    print("\n" + "=" * 60)
    print("2. 蒙特卡洛积分计算 ∫₀¹ e⁻ˣ dx")
    print("=" * 60)
    
    # 精确积分值
    exact_value, _ = quad(f, 0, 1)
    
    n_trials = 10000
    
    # 随机投点法
    hm_estimate, hit_xs, hit_ys, miss_xs, miss_ys = monte_carlo_hit_or_miss(n_trials)
    
    # 样本平均法
    sm_estimate, samples, func_vals = monte_carlo_sample_mean(n_trials)
    
    print(f"\n精确积分值: {exact_value:.6f}")
    print(f"随机投点法估计值: {hm_estimate:.6f}")
    print(f"绝对误差: {abs(hm_estimate - exact_value):.6f}")
    print(f"样本平均法估计值: {sm_estimate:.6f}")
    print(f"绝对误差: {abs(sm_estimate - exact_value):.6f}")
    
    # 画出该定积分的面积区域
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：随机投点法
    ax1 = axes[0]
    x_curve = np.linspace(0, 1, 1000)
    y_curve = f(x_curve)
    
    ax1.plot(x_curve, y_curve, 'r-', linewidth=2, label=r'$f(x) = e^{-x}$')
    ax1.fill_between(x_curve, 0, y_curve, alpha=0.3, color='red', label='Integral Area')
    
    # 画点（只画部分避免过于密集）
    sample_size = min(2000, len(hit_xs))
    ax1.scatter(hit_xs[:sample_size], hit_ys[:sample_size], 
               c='green', s=5, alpha=0.5, label='Hit Points')
    ax1.scatter(miss_xs[:sample_size], miss_ys[:sample_size], 
               c='blue', s=5, alpha=0.5, label='Miss Points')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Hit-or-Miss Method (n={n_trials})\nEstimate = {hm_estimate:.4f}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图：样本平均法
    ax2 = axes[1]
    ax2.plot(x_curve, y_curve, 'r-', linewidth=2, label=r'$f(x) = e^{-x}$')
    ax2.fill_between(x_curve, 0, y_curve, alpha=0.3, color='red', label='Integral Area')
    
    # 画采样点
    sample_size = min(500, len(samples))
    ax2.scatter(samples[:sample_size], func_vals[:sample_size], 
               c='purple', s=10, alpha=0.5, label='Sample Points')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title(f'Sample Mean Method (n={n_trials})\nEstimate = {sm_estimate:.4f}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(r'Monte Carlo Simulation of $\int_0^1 e^{-x} dx$', fontsize=14)
    plt.tight_layout()
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 蒲丰投针问题
    buffon_needle_experiments()
    
    # 2. 蒙特卡洛积分
    integral_experiment()