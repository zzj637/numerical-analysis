import numpy as np
import matplotlib.pyplot as plt
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 微分方程右端函数
def f(x, y):
    return -1/x**2 - y/x - y**2

# 精确解
def y_exact(x):
    # y = -tan(ln(x) + pi/4) / x
    return -np.tan(np.log(x) + np.pi/4) / x

# --------------------- 数值方法 ---------------------
def euler_step(x, y, h):
    return y + h * f(x, y)

def improved_euler_step(x, y, h):
    y_p = euler_step(x, y, h)
    return y + h/2 * (f(x, y) + f(x+h, y_p))

def heun2_step(x, y, h):
    K1 = f(x, y)
    K2 = f(x + 2/3*h, y + 2/3*h*K1)
    return y + h/4 * (K1 + 3*K2)

def midpoint_step(x, y, h):
    return y + h * f(x + h/2, y + h/2 * f(x, y))

def rk4_step(x, y, h):
    K1 = f(x, y)
    K2 = f(x + h/2, y + h/2 * K1)
    K3 = f(x + h/2, y + h/2 * K2)
    K4 = f(x + h,   y + h   * K3)
    return y + h/6 * (K1 + 2*K2 + 2*K3 + K4)

# --------------------- 主计算 ---------------------
def solve_ivp(step_method, x0, y0, h, N):
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0], y[0] = x0, y0
    for i in range(N):
        y[i+1] = step_method(x[i], y[i], h)
        x[i+1] = x[i] + h
    return x, y

x0, y0 = 1.0, -1.0
h = 0.1
N = 10

methods = {
    "Euler":          euler_step,
    "改进Euler":       improved_euler_step,
    "Heun":           heun2_step,
    "中点":           midpoint_step,
    "RK4":            rk4_step
}

results = {}
for name, method in methods.items():
    x_vals, y_vals = solve_ivp(method, x0, y0, h, N)
    results[name] = (x_vals, y_vals)

# 精确解
x_dense = np.linspace(1, 2, 200)
y_dense = y_exact(x_dense)

# --------------------- 表格输出 ---------------------
print("\n节点 & 精确解", end="")
for name in methods:
    print(f" & {name}数值解 & {name}误差", end="")
print(" \\\\")
print("\\hline")

for i in range(N+1):
    x_i = 1.0 + i*h
    y_e = y_exact(x_i)
    print(f"{x_i:.1f} & {y_e:.6f}", end="")
    for name in methods:
        y_n = results[name][1][i]
        err = abs(y_n - y_e)
        # 根据误差大小科学计数
        if err == 0:
            err_str = "0.0e+00"
        else:
            err_str = f"{err:.1e}"
            # 格式化指数为两位数字
        print(f" & {y_n:.6f} & {err_str}", end="")
    print(" \\\\")

# --------------------- 绘图 ---------------------
plt.figure(figsize=(12, 7))
plt.plot(x_dense, y_dense, 'k-', linewidth=2, label='精确解')

colors = ['blue', 'red', 'green', 'orange', 'purple']
markers = ['o', 's', 'D', '^', 'v']
for (name, (x, y)), c, m in zip(results.items(), colors, markers):
    plt.plot(x, y, marker=m, color=c, linestyle='--', label=name, markersize=5)

plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.title('常微分方程初值问题数值解比较 ($h=0.1$)', fontsize=15)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=200)
plt.show()