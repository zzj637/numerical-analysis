import numpy as np
import matplotlib.pyplot as plt

# 原函数及其导数
def R(x):
    return 1.0 / (1.0 + x * x)

def dR(x):
    return -2.0 * x / (1.0 + x * x) ** 2

# 追赶法求解三对角方程组 Ax = d
def solve_tridiagonal(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i-1]
        if i < n-1:
            cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i-1]) / m
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

# 三次自然样条插值类
class NaturalSpline:
    def __init__(self, x_nodes, y_nodes):
        self.x = np.array(x_nodes)
        self.y = np.array(y_nodes)
        self.n = len(x_nodes) - 1
        self.h = x_nodes[1] - x_nodes[0]   # 等距节点，步长固定

        # 构造三对角方程组 (内部节点数 = n-1)
        m = self.n - 1
        a = np.ones(m)          # 下对角线
        b = 4.0 * np.ones(m)    # 主对角线
        c = np.ones(m)          # 上对角线
        d = np.zeros(m)
        for i in range(1, self.n):
            d[i-1] = 6.0 / (self.h * self.h) * (self.y[i-1] - 2*self.y[i] + self.y[i+1])

        # 求解内部节点的二阶导数 M1...M_{n-1}
        m_inner = solve_tridiagonal(a, b, c, d)
        # 边界条件 M0 = Mn = 0 (自然样条)
        self.M = np.concatenate(([0.0], m_inner, [0.0]))

    def eval(self, x):
        # 确定 x 所在的区间下标
        if x == self.x[-1]:
            i = self.n - 1
        else:
            i = int(np.floor((x - self.x[0]) / self.h))
        i = max(0, min(i, self.n - 1))

        xi = self.x[i]
        xi1 = self.x[i+1]
        hi = self.h
        Mi = self.M[i]
        Mi1 = self.M[i+1]
        yi = self.y[i]
        yi1 = self.y[i+1]

        t1 = xi1 - x
        t2 = x - xi
        term1 = (Mi * t1**3 + Mi1 * t2**3) / (6 * hi)
        term2 = (yi / hi - Mi * hi / 6) * t1
        term3 = (yi1 / hi - Mi1 * hi / 6) * t2
        return term1 + term2 + term3

# 分段三次 Hermite 插值 (使用基函数形式)
def hermite_eval(x, xn, yn, ypn):
    h = xn[1] - xn[0]   # 等距步长
    if x == xn[-1]:
        i = len(xn) - 2
    else:
        i = int(np.floor((x - xn[0]) / h))
    i = max(0, min(i, len(xn)-2))

    t = (x - xn[i]) / h
    # Hermite 基函数插值
    return (yn[i] * (1 + 2*t) * (1-t)**2 +
            yn[i+1] * (3 - 2*t) * t**2 +
            ypn[i] * h * t * (1-t)**2 +
            ypn[i+1] * h * (t-1) * t**2)

# 主程序
def main():
    # 等距节点 x_i = -5 + i, i = 0..10
    xi = np.array([-5.0 + i for i in range(11)])
    yi = R(xi)
    ypi = dR(xi)

    # 创建自然样条对象
    spline = NaturalSpline(xi, yi)

    # 生成绘图所需的密集点
    x_plot = np.linspace(-5, 5, 1000)
    y_original = R(x_plot)
    y_spline = np.array([spline.eval(x) for x in x_plot])
    y_hermite = np.array([hermite_eval(x, xi, yi, ypi) for x in x_plot])

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_original, 'k-', linewidth=2, label='$R(x)=1/(1+x^2)$')
    plt.plot(x_plot, y_spline, 'r--', linewidth=1.5, label='三次自然样条插值')
    plt.plot(x_plot, y_hermite, 'b-.', linewidth=1.5, label='分段三次 Hermite 插值')
    plt.scatter(xi, yi, c='green', s=30, zorder=5, label='插值节点')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('三次自然样条与分段三次 Hermite 插值对比')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison.png', dpi=300)
    plt.show()

    # 可选：输出部分数值结果（与原 C++ 代码类似）
    print("x\tOriginal\tSpline\t\tHermite")
    for x in np.arange(-5, 5.1, 0.5):
        print(f"{x:.6f}\t{R(x):.6f}\t{spline.eval(x):.6f}\t{hermite_eval(x, xi, yi, ypi):.6f}")

if __name__ == "__main__":
    main()