import math
import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ---------- 插值算法实现（与之前相同） ----------
def R(x):
    """原函数 R(x) = 1/(1+x^2)"""
    return 1.0 / (1.0 + x * x)

def lagrange(x, xi, yi):
    """拉格朗日插值"""
    n = len(xi)
    result = 0.0
    for i in range(n):
        term = yi[i]
        for j in range(n):
            if i != j:
                term *= (x - xi[j]) / (xi[i] - xi[j])
        result += term
    return result

def newton(x, xi, yi):
    """牛顿插值（差商表 + Horner）"""
    n = len(xi)
    f = [[0.0] * n for _ in range(n)]
    for i in range(n):
        f[i][0] = yi[i]
    for j in range(1, n):
        for i in range(j, n):
            f[i][j] = (f[i][j-1] - f[i-1][j-1]) / (xi[i] - xi[i-j])
    res = f[n-1][n-1]
    for i in range(n-2, -1, -1):
        res = f[i][i] + (x - xi[i]) * res
    return res

def piecewise_linear(x, xi, yi):
    """分段线性插值"""
    n = len(xi)
    if x <= xi[0]:
        return yi[0]
    if x >= xi[-1]:
        return yi[-1]
    for i in range(n-1):
        if xi[i] <= x <= xi[i+1]:
            return yi[i] + (yi[i+1] - yi[i]) * (x - xi[i]) / (xi[i+1] - xi[i])
    return 0.0

# ---------- 生成节点 ----------
# 1. 切比雪夫节点（用于20次Lagrange插值）
PI = math.acos(-1.0)
x_cheb = [5.0 * math.cos((2.0*i + 1.0) / 42.0 * PI) for i in range(21)]
y_cheb = [R(x) for x in x_cheb]

# 2. 等距节点（用于Newton和分段线性插值）
x_eq = [-5.0 + i for i in range(11)]
y_eq = [R(x) for x in x_eq]

# ---------- 生成绘图数据 ----------
x_plot = np.linspace(-5, 5, 1000)          # 1000个点用于平滑绘图
y_true = R(x_plot)
y_lagrange = [lagrange(x, x_cheb, y_cheb) for x in x_plot]
y_newton = [newton(x, x_eq, y_eq) for x in x_plot]
y_piecewise = [piecewise_linear(x, x_eq, y_eq) for x in x_plot]

# ---------- 绘图 ----------
plt.figure(figsize=(12, 7))

# 原函数
plt.plot(x_plot, y_true, 'k-', linewidth=2, label='原函数 $R(x)=1/(1+x^2)$')

# 20次Lagrange插值（切比雪夫节点）
plt.plot(x_plot, y_lagrange, 'b--', linewidth=1.5, label='20次Lagrange插值 (切比雪夫节点)')

# 10次Newton插值（等距节点，Runge现象）
plt.plot(x_plot, y_newton, 'r:', linewidth=1.5, label='10次Newton插值 (等距节点)')

# 分段线性插值
plt.plot(x_plot, y_piecewise, 'g-.', linewidth=1.5, label='分段线性插值')

# 标记节点（可选，增强可读性）
plt.plot(x_cheb, y_cheb, 'bo', markersize=3, label='切比雪夫节点')
plt.plot(x_eq, y_eq, 'rs', markersize=3, label='等距节点')

# 设置坐标轴范围（突出Runge现象，同时保证其他曲线可见）
plt.xlim(-5.5, 5.5)
plt.ylim(-0.5, 1.5)          # 牛顿插值在边界处会剧烈震荡，限制y轴范围以观察全貌

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('三种插值方法与原函数 $R(x)$ 的对比', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# 添加文字说明Runge现象和切比雪夫节点效果
plt.text(3.5, 1.3, 'Runge现象\n（等距节点高次震荡）', ha='center', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.text(-4.5, 0.2, '切比雪夫节点\n抑制震荡', ha='center', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 保存图像（与LaTeX文档中引用的文件名一致）
plt.savefig('plot_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("图像已保存为 plot_results.png")