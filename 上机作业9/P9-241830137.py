import numpy as np

def f32(x):
    """单精度被积函数"""
    return np.float32(4.0) / (np.float32(1.0) + np.float32(x)**2)

def composite_trapezoidal_float32(a, b, n):
    """复合梯形公式，float32单精度"""
    h = np.float32((b - a) / n)
    s = np.float32(0.5) * f32(a) + np.float32(0.5) * f32(b)
    for i in range(1, n):
        s += f32(a + np.float32(i) * h)
    return float(h * s)

def composite_simpson_float32(a, b, n):
    """复合Simpson公式，float32单精度"""
    if n % 2 != 0:
        n += 1
    h = np.float32((b - a) / n)
    s = f32(a) + f32(b)
    for i in range(1, n, 2):
        s += np.float32(4.0) * f32(a + np.float32(i) * h)
    for i in range(2, n, 2):
        s += np.float32(2.0) * f32(a + np.float32(i) * h)
    return float(h / np.float32(3.0) * s)


a, b = 0, 1
exact = np.pi

print("使用float32单精度，误差不再减小的现象更显著：\n")
print(f"{'n':<10} {'h':<14} {'梯形误差':<16} {'有改进?':<10} {'Simpson误差':<16} {'有改进?':<10}")
print("-" * 76)

prev_T = None
prev_S = None

for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    h = (b - a) / n
    err_T = abs(composite_trapezoidal_float32(a, b, n) - exact)
    err_S = abs(composite_simpson_float32(a, b, n) - exact)

    if prev_T is None:
        T_better, S_better = "—", "—"
    else:
        T_better = "是" if err_T < prev_T else "否"
        S_better = "是" if err_S < prev_S else "否"

    print(f"{n:<10} {h:<14.6e} {err_T:<16.6e} {T_better:<10} {err_S:<16.6e} {S_better:<10}")

    prev_T = err_T
    prev_S = err_S

