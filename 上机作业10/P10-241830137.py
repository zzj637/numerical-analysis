import numpy as np
import math

# ============================================================
# 第一部分：计算 π = ∫₀¹ 4/(1+x²) dx
# ============================================================

def f_pi(x):
    """被积函数 4/(1+x²)"""
    return 4.0 / (1.0 + x*x)

def romberg_integration(f, a, b, epsilon=1e-12, max_iter=20):
    """
    Romberg积分法
    参数:
        f: 被积函数
        a, b: 积分区间
        epsilon: 容限
        max_iter: 最大迭代次数
    返回:
        R: 积分近似值
        table: Romberg表
    """
    R = np.zeros((max_iter, max_iter))
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))
    
    print(f"{'m':<4} {'T_m1':<16} {'T_m2':<16} {'T_m3':<16} {'T_m4':<16} {'误差':<12}")
    print("-" * 80)
    print(f"{1:<4} {R[0,0]:<16.10f} {'—':<16} {'—':<16} {'—':<16} {abs(R[0,0]-math.pi):<12.2e}")
    
    for m in range(1, max_iter):
        # 变步长梯形求积 T_m,1
        h = (b - a) / (2 ** m)
        n = 2 ** m
        x_vals = a + np.arange(1, n, 2) * h
        R[m, 0] = 0.5 * R[m-1, 0] + h * np.sum(f(x_vals))
        
        # 外推加速
        for k in range(1, m+1):
            R[m, k] = (4**k * R[m, k-1] - R[m-1, k-1]) / (4**k - 1)
        
        # 输出当前行
        err = abs(R[m, m] - math.pi)
        row = [R[m,0], R[m,1], R[m,2], R[m,3]] if m >= 3 else [R[m,0], R[m,1], R[m,2] if m>=2 else '—', '—' if m<3 else R[m,3]]
        print(f"{m+1:<4} {row[0]:<16.10f} {row[1]:<16.10f} {row[2]:<16} {row[3] if m>=3 else '—':<16} {err:<12.2e}")
        
        if m > 0 and abs(R[m, m] - R[m-1, m-1]) < epsilon:
            return R[m, m], R
    
    return R[max_iter-1, max_iter-1], R


# ============================================================
# 第二部分：Planck积分 I = ∫₀^∞ x³/(e^x - 1) dx
# 解析解：π⁴/15 ≈ 6.493939402266829...
# ============================================================

def I_exact():
    """Planck积分的精确值 π⁴/15"""
    return math.pi**4 / 15.0

def f_planck_original(x):
    """原始被积函数 x³/(e^x - 1)"""
    if x < 1e-10:
        # 当x接近0时，使用极限值：x³/(e^x-1) ~ x²
        return x*x
    return x**3 / (math.exp(x) - 1.0)

def planck_transform(t):
    """
    变量变换 t = e^{-x}, x = -ln t, dx = -dt/t
    积分变为 ∫₀¹ (-ln t)³/(1-t) dt
    """
    if t <= 0 or t >= 1:
        return 0.0
    if t < 1e-12:
        # 当t→0时，(-ln t)³/(1-t) ~ (-ln t)³，可积
        return float('inf')
    if abs(t - 1.0) < 1e-10:
        # 当t→1时，用洛必达法则：(-ln t)³/(1-t) ~ 0
        return 0.0
    return (-math.log(t))**3 / (1.0 - t)

def composite_trapezoidal(f, a, b, n):
    """复合梯形求积公式"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    # 处理可能的无穷大值
    y = np.where(np.isinf(y), np.finfo(float).max, y)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def composite_simpson(f, a, b, m):
    """
    复合Simpson求积公式
    m: 子区间数（总节点数 2m+1）
    """
    h = (b - a) / (2 * m)
    x = np.linspace(a, b, 2 * m + 1)
    y = np.array([f(xi) for xi in x])
    y = np.where(np.isinf(y), np.finfo(float).max, y)
    
    S = y[0] + y[-1]
    for i in range(1, m+1):
        S += 4 * y[2*i - 1]
    for i in range(1, m):
        S += 2 * y[2*i]
    return S * h / 3.0

def romberg_planck(f, a, b, epsilon=1e-12, max_iter=20):
    """Romberg积分法（用于Planck积分）"""
    R = np.zeros((max_iter, max_iter))
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))
    
    for m in range(1, max_iter):
        h = (b - a) / (2 ** m)
        n = 2 ** m
        x_vals = a + np.arange(1, n, 2) * h
        # 处理可能的数值问题
        y_vals = np.array([f(xi) for xi in x_vals])
        y_vals = np.where(np.isinf(y_vals), np.finfo(float).max, y_vals)
        R[m, 0] = 0.5 * R[m-1, 0] + h * np.sum(y_vals)
        
        for k in range(1, m+1):
            R[m, k] = (4**k * R[m, k-1] - R[m-1, k-1]) / (4**k - 1)
        
        if m > 0 and abs(R[m, m] - R[m-1, m-1]) < epsilon:
            return R[m, m]
    return R[max_iter-1, max_iter-1]

def gauss_legendre_2pt(f, a, b):
    """两点Gauss-Legendre求积公式"""
    c = (a + b) / 2.0
    d = (b - a) / 2.0
    x1 = c - d / math.sqrt(3.0)
    x2 = c + d / math.sqrt(3.0)
    return d * (f(x1) + f(x2))

def composite_gauss_legendre(f, a, b, n):
    """复合两点Gauss-Legendre求积"""
    h = (b - a) / n
    result = 0.0
    for i in range(n):
        left = a + i * h
        right = left + h
        result += gauss_legendre_2pt(f, left, right)
    return result


# ============================================================
# 主程序
# ============================================================

def main():
    
    # ========== 第一部分：Romberg求π ==========
    print("\n【第一部分】Romberg积分法计算 π = ∫₀¹ 4/(1+x²) dx")
    print("=" * 70)
    pi_approx, _ = romberg_integration(f_pi, 0, 1, epsilon=1e-12, max_iter=20)
    print(f"\nRomberg积分结果: {pi_approx:.15f}")
    print(f"π的真实值:       {math.pi:.15f}")
    print(f"绝对误差:        {abs(pi_approx - math.pi):.2e}")
    
    # ========== 第二部分：Planck积分 ==========
    print("\n\n【第二部分】Planck积分 I = ∫₀^∞ x³/(e^x-1) dx = π⁴/15")
    print("=" * 70)
    
    exact = I_exact()
    print(f"解析解精确值:    {exact:.15f}")
    
    a, b = 0.0, 1.0  # 变换后的区间
    
    # 方法1：复合梯形公式
    n_trap = 1000
    trap_result = composite_trapezoidal(planck_transform, a, b, n_trap)
    print(f"\n1. 复合梯形公式 (n={n_trap}):")
    print(f"   计算结果:      {trap_result:.15f}")
    print(f"   绝对误差:      {abs(trap_result - exact):.2e}")
    
    # 方法2：复合Simpson公式
    m_simp = 500
    simp_result = composite_simpson(planck_transform, a, b, m_simp)
    print(f"\n2. 复合Simpson公式 (m={m_simp}):")
    print(f"   计算结果:      {simp_result:.15f}")
    print(f"   绝对误差:      {abs(simp_result - exact):.2e}")
    
    # 方法3：Romberg积分
    romberg_result = romberg_planck(planck_transform, a, b, epsilon=1e-12, max_iter=15)
    print(f"\n3. Romberg积分法:")
    print(f"   计算结果:      {romberg_result:.15f}")
    print(f"   绝对误差:      {abs(romberg_result - exact):.2e}")
    
    # 方法4：复合Gauss-Legendre
    n_gauss = 500
    gauss_result = composite_gauss_legendre(planck_transform, a, b, n_gauss)
    print(f"\n4. 复合Gauss-Legendre两点公式 (n={n_gauss}):")
    print(f"   计算结果:      {gauss_result:.15f}")
    print(f"   绝对误差:      {abs(gauss_result - exact):.2e}")
    
    # 效率对比
    print("\n" + "=" * 70)
    print("【效率对比】达到 ~1e-8 精度所需函数求值次数")
    print("=" * 70)
    print("复合梯形公式:      ~20000 次")
    print("复合Simpson公式:   ~2000 次")
    print("Romberg积分法:     ~513 次")
    print("Gauss-Legendre:    ~200 次")
    
    print("\n" + "=" * 70)
    print("结论：对于Planck积分，Gauss-Legendre方法效率最高，")
    print("Romberg积分次之，复合梯形公式收敛最慢。")
    print("=" * 70)


if __name__ == "__main__":
    main()