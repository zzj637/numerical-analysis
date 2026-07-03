import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from datetime import datetime, timedelta

def optimize(f, df, x0, eta, eps=1e-8):
    """梯度下降法优化器"""
    n = len(x0)
    x = x0.copy().astype(float)
    k = 0
    max_iter = 50000
    
    while k < max_iter:
        g = np.array([df[i](x) for i in range(n)])
        if np.linalg.norm(g) < eps:
            print(f"  收敛于第{k}步, ||g||={np.linalg.norm(g):.2e}")
            break
        x = x - eta * g
        k += 1
    
    if k == max_iter:
        print(f"  达到最大迭代次数{max_iter}")
    return x

def logistic_model(t, P0, K, r):
    """Logistic模型: P(t) = K / (1 + (K/P0-1)*exp(-r*t))"""
    return K / (1 + (K/P0 - 1) * np.exp(-r * t))

def solve_logistic(t_data, P_data, P0_init, K_init, r_init, eta=1e-8):
    """拟合Logistic模型的三个参数(P0, K, r)"""
    
    def loss(params):
        P0, K, r = params
        pred = logistic_model(t_data, P0, K, r)
        return np.mean((pred - P_data)**2)
    
    def dloss_dP0(params):
        P0, K, r = params
        pred = logistic_model(t_data, P0, K, r)
        d = (1 + (K/P0 - 1)*np.exp(-r*t_data))**2
        dpred = (K**2/P0**2)*np.exp(-r*t_data)/d
        return 2*np.mean((pred - P_data)*dpred)
    
    def dloss_dK(params):
        P0, K, r = params
        pred = logistic_model(t_data, P0, K, r)
        d = (1 + (K/P0 - 1)*np.exp(-r*t_data))**2
        dpred = 1/(1 + (K/P0 - 1)*np.exp(-r*t_data)) - \
                K*(1/P0)*np.exp(-r*t_data)/d
        return 2*np.mean((pred - P_data)*dpred)
    
    def dloss_dr(params):
        P0, K, r = params
        pred = logistic_model(t_data, P0, K, r)
        d = (1 + (K/P0 - 1)*np.exp(-r*t_data))**2
        dpred = K*(K/P0 - 1)*t_data*np.exp(-r*t_data)/d
        return 2*np.mean((pred - P_data)*dpred)
    
    x0 = np.array([P0_init, K_init, r_init])
    df_list = [dloss_dP0, dloss_dK, dloss_dr]
    x_opt = optimize(loss, df_list, x0, eta)
    return x_opt


if __name__ == "__main__":
    # 读取数据
    d = read_csv('wuhan2020.csv')
    df = d.values
    dates = [datetime.strptime(str(dd).strip(), '%Y/%m/%d') 
             for dd in df[:, 0]]
    cases = df[:, 1].astype(float)
    
    base_date = datetime(2020, 1, 16)
    t = np.array([(d - base_date).days for d in dates])
    P = cases
    
    # ===== 初始值设定 =====
    P0_init = cases[0]
    K_init = cases[-1] * 1.05
    r_init = 0.3
    
    print("=" * 60)
    print("COVID-19 武汉市 Logistic 模型拟合")
    print("=" * 60)
    
    # (1) 全数据拟合
    print("\n(1) 全数据拟合 (1/16 - 3/16)")
    opt_f = solve_logistic(t, P, P0_init, K_init, r_init, eta=1e-10)
    P0_f, K_f, r_f = opt_f
    print(f"  P0={P0_f:.2f}, K={K_f:.0f}, r={r_f:.4f}")
    
    t_inf_f = np.log(K_f/P0_f - 1)/r_f
    inf_date_f = base_date + timedelta(days=t_inf_f)
    print(f"  拐点: {t_inf_f:.2f}天 -> {inf_date_f.strftime('%Y/%m/%d')}")
    
    # (2) 早期数据拟合
    print("\n(2) 早期数据拟合 (1/16 - 1/31)")
    cutoff = 16
    t_e, P_e = t[:cutoff], P[:cutoff]
    
    opt_e = solve_logistic(t_e, P_e, P0_init, K_init, r_init, eta=1e-10)
    P0_e, K_e, r_e = opt_e
    print(f"  P0={P0_e:.2f}, K={K_e:.0f}, r={r_e:.4f}")
    
    t_inf_e = np.log(K_e/P0_e - 1)/r_e
    inf_date_e = base_date + timedelta(days=t_inf_e)
    print(f"  拐点: {t_inf_e:.2f}天 -> {inf_date_e.strftime('%Y/%m/%d')}")
    
    # 计算预测误差
    P_pred_e = logistic_model(t, P0_e, K_e, r_e)
    mre = np.mean(np.abs((P_pred_e[cutoff:] - P[cutoff:])/P[cutoff:]))*100
    print(f"  2月1日后预测平均相对误差: {mre:.2f}%")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    t_fine = np.linspace(0, max(t), 300)
    
    ax1.scatter(t, P, s=15, c='black', zorder=5, label='Real')
    ax1.plot(t_fine, logistic_model(t_fine, *opt_f), 
             'r-', lw=2, label='Full data')
    ax1.plot(t_fine, logistic_model(t_fine, *opt_e), 
             'b--', lw=2, label='Early data')
    ax1.axvline(x=cutoff-1, color='gray', ls=':', label='Cut-off')
    ax1.axvline(x=t_inf_f, color='red', ls='-.', label='Inflection(full)')
    ax1.axvline(x=t_inf_e, color='blue', ls='-.', label='Inflection(early)')
    ax1.set_xlabel('Days since 2020/1/16')
    ax1.set_ylabel('Confirmed Cases')
    ax1.set_title('Logistic Model Fit')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(t, P, s=15, c='black', zorder=5, label='Real')
    ax2.plot(t, P_pred_e, 'b--', lw=2, label='Early prediction')
    ax2.plot(t, logistic_model(t, *opt_f), 'r-', lw=2, label='Full fit')
    ax2.axvline(x=cutoff-1, color='gray', ls=':', label='Cut-off')
    ax2.set_xlabel('Days since 2020/1/16')
    ax2.set_ylabel('Confirmed Cases')
    ax2.set_title('Early Data Prediction')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('covid_logistic.png', dpi=150)
    plt.show()