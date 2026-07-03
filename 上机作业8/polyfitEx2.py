import numpy as np
import matplotlib.pyplot as plt

def polyfit(xdata, ydata, f): 
    # 给定xdata和ydata, 函数f, 返回广义多项式. 
    # f可以输入数字，如果输入数字，那么就表示多项式的次数. 
    # f也可以输入函数数组f=[f1, ..., f_n]，表示广义多项式拟合，同时加上常值函数. 
    m = len(xdata)

    if(type(f)==int): #多项式
        n = f + 1 #f表示deg
        xdatas = np.zeros([n,m])
        xdatas[0] = np.ones(m)
        for i in range(1,n):    # 用不断迭代的方法得到xdata的幂次.
            xdatas[i] = xdatas[i-1] * xdata 
    else:
        n = len(f) + 1 
        xdatas = np.zeros([n,m])
        xdatas[0] = np.ones(m)
        for i in range(1, n):   
            # TODO: 请补充完整这里
            xdatas[i] = f[i-1](xdata)

    # 装配矩阵 A 
    A = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            A[i][j] = np.dot(xdatas[n-1-i],xdatas[n-1-j])

    # 装配右端向量 b
    b = np.zeros(n)
    for i in range(n):
        b[i] = np.dot(xdatas[n-1-i],ydata)

    # 解方程组
    a = np.linalg.solve(A, b)
    return a

# 调用

if __name__ == "__main__":
    xdata = np.array([0.000, 0.895, 1.641, 2.512, 3.542, 4.054, 4.602, 5.063, 5.354, 5.617])
    ydata = np.array([1.000, 1.803, 3.680, 7.320, 13.59, 17.41, 22.19, 24.89, 26.55, 29.77])
    m = len(ydata)
    
    def f0(x):
        return x
    def f1(x):
        return np.cos(x) 
    def f2(x):
        return np.sin(x)

    a1 = polyfit(xdata,ydata,2)  #polyfit返回值是次数从大到小的拟合多项式的系数.
    print(a1)

    f = [f0, f1, f2]
    a2 = polyfit(xdata,ydata,f)  #polyfit返回值是次数从大到小的拟合多项式的系数.
    print(a2)
    # [-1.90183422  2.68270259  4.90973928 -2.15461946]
    # -1.9f0 + 2.68f1 + 4.91f2 - 2.15