import numpy as np
import matplotlib.pyplot as plt

def optimize(f, df, x0, eta, eps=1e-8):
    """
    梯度下降法优化器
    输入: f  - 目标函数 f(x1, x2, ..., xn)
          df - 梯度函数列表 [df1, df2, ..., dfn]
          x0 - 初始点
          eta - 步长
          eps - 收敛容限
    输出: (最优解, 损失历史)
    """
    n = len(x0)
    # 检查输入维数
    assert f.__code__.co_argcount == n, \
        'Dimension of input of f does not match x0'
    assert len(df) == n, \
        'Dimension of df does not match x0'
    
    x = x0.copy().astype(float)
    k = 0
    max_iter = 10000
    history = []
    
    while k < max_iter:
        # 计算梯度向量
        g = np.array([df[i](*x) for i in range(n)])
        history.append(f(*x))
        
        # 收敛判定
        if np.linalg.norm(g) < eps:
            print(f"梯度下降在第{k}步收敛, ||g||={np.linalg.norm(g):.2e}")
            break
        
        # 沿负梯度方向更新
        x = x - eta * g
        k += 1
    
    if k == max_iter:
        print(f"达到最大迭代次数{max_iter}")
    
    return x, np.array(history)


if __name__ == "__main__":
    # ===== 验证算例 =====
    print("=" * 50)
    print("验证梯度下降法: f(x1, x2) = x1^2 + x2^2")
    print("=" * 50)
    
    def f_test(x1, x2):
        return x1*x1 + x2*x2
    def df1_test(x1, x2):
        return 2*x1
    def df2_test(x1, x2):
        return 2*x2
    
    x0 = np.array([0.5, 0.5])
    eta = 0.1
    x_opt, hist = optimize(f_test, [df1_test, df2_test], x0, eta)
    print(f"最优解: ({x_opt[0]:.2e}, {x_opt[1]:.2e})")
    print(f"最优值: {f_test(*x_opt):.2e}")
    print(f"迭代步数: {len(hist)}")
    
    # ===== 非线性数据拟合 =====
    print("\n" + "=" * 50)
    print("非线性拟合: y = 2/(1 + a*exp(b*x))")
    print("=" * 50)
    
    xdata = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    ydata = np.array([0.04, 0.17, 0.65, 1.39, 1.85, 1.90, 1.99])
    
    # 线性化技巧获取初值
    Y = np.log(2.0/ydata - 1.0)
    A = np.vstack([np.ones_like(xdata), xdata]).T
    c, b0 = np.linalg.lstsq(A, Y, rcond=None)[0]
    a0 = np.exp(c)
    print(f"线性化初值: a={a0:.4f}, b={b0:.4f}")
    
    # 定义损失函数及其梯度
    def loss(a, b):
        pred = 2.0 / (1 + a*np.exp(b*xdata))
        return np.mean((pred - ydata)**2)
    
    def dloss_da(a, b):
        pred = 2.0 / (1 + a*np.exp(b*xdata))
        dpred = -2*np.exp(b*xdata) / (1 + a*np.exp(b*xdata))**2
        return 2*np.mean((pred - ydata)*dpred)
    
    def dloss_db(a, b):
        pred = 2.0 / (1 + a*np.exp(b*xdata))
        dpred = -2*a*xdata*np.exp(b*xdata) / (1 + a*np.exp(b*xdata))**2
        return 2*np.mean((pred - ydata)*dpred)
    
    # 梯度下降优化
    x0 = np.array([a0, b0])
    eta = 0.01
    x_opt, history = optimize(loss, [dloss_da, dloss_db], x0, eta)
    
    a_opt, b_opt = x_opt
    print(f"优化结果: a={a_opt:.4f}, b={b_opt:.4f}")
    print(f"初始损失: {loss(a0, b0):.2e}")
    print(f"最终损失: {loss(a_opt, b_opt):.2e}")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    x_fine = np.linspace(-1.5, 1.5, 200)
    y_fit = 2.0/(1 + a_opt*np.exp(b_opt*x_fine))
    ax1.scatter(xdata, ydata, c='red', s=40, zorder=5, label='Data')
    ax1.plot(x_fine, y_fit, 'b-', lw=2, label='Fitted')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(r'$y = 2/(1 + a e^{bx})$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss History')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonlinfit_result.png', dpi=150)
    plt.show()