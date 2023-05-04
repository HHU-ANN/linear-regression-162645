# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import os
#import numpy
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    #读取数据
    X,y = read_data()
    #weight = np.ones((6,1))
    '''print(np.shape(X))
    I = np.eye(6)
    #print(0.1*I)
    c = 1
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)+c*I),np.matmul(X.T,y))
    return weight @ data'''
    # 添加多项式特征
    X_poly = X
    alpha = 1e-12
    degree = 1
    for d in range(2, degree + 1):
        X_poly = np.concatenate((X_poly, np.power(X, d)), axis=1)

    # 求解岭回归问题
    m, n = X_poly.shape
    I = np.eye(n)
    w = np.linalg.inv(X_poly.T @ X_poly + alpha * I) @ X_poly.T @ y

    return w @ data


def sfun(t,ld):
    tmp = (np.abs(t)-ld)
    if tmp < 0:
        tmp = 0
    return np.sign(t)**tmp
def lasso(data):
    '''X, y = read_data()
    #初始化权重
    weight = np.zeros(X.shape[1])
    number = y.size
    for i in range(10):
        #计算预测值与真实值之间的误差
        error = X.dot(weight) - y
        #计算梯度
        l1_ratio = 0.9
        l1_grad = l1_ratio * np.sign(weight)
        ls_grad = (1 - l1_ratio) * (X.T.dot(error)) / number
        #求梯度之和
        grad = ls_grad + l1_grad
        #设置更新步长
        alpha = 0.1
        #更新权重
        weight -= alpha * grad
        #print(weight)
    return weight @ data'''
    #参数设置
    '''iternum = 1000
    lamda = 0.1
    X, y = read_data('D:/myfile/data/shenjingwangluo/linear-regression-162645/data/exp02/')
    m, n = X.shape
    theta = np.matrix(np.zeros((n, 1)))
    costs = np.zeros(iternum)
    # 循环
    for it in range(iternum):
        for k in range(n):  # n个特征
            # 计算z_k和p_k
            z_k = np.sum(np.power(X[:, k], 2))
            p_k = 0
            for i in range(m):
                p_k += X[i, k] * (y[i, 0] - np.sum([X[i, j] * theta[j, 0] for j in range(n) if j != k]))
            # 根据p_k的不同取值进行计算
            if p_k < -lamda / 2:
                w_k = (p_k + lamda / 2) / z_k
            elif p_k > lamda / 2:
                w_k = (p_k - lamda / 2) / z_k
            else:
                w_k = 0
            theta[k, 0] = w_k
        costs[it] = np.sum(np.power((np.dot(X, theta) - y), 2)) / (2 * m) + lamda * np.sum(np.abs(theta))'''
    '''X, y = read_data('D:/myfile/data/shenjingwangluo/linear-regression-162645/data/exp02/')
    n, m = X.shape
    omega = np.ones([m, 1])
    # 外层迭代迭代总搜索次数
    for i in range(epochs):
        # preomega表示上次搜索的omega
        pre_omega = copy.copy(omega)
        for j in range(m):
            # 内层迭代迭代每个维度j的搜索次数
            for k in range(epochs):
                yhat = xmat * omega
                temp = xmat[:, j].T * (yhat - ymat) / n + lam * np.sign(omega[j])
                omega[j] = omega[j] - temp * lr
                # 若该omega的第j个维度已经足够接近0则终止内层迭代
                if np.abs(omega[j]) < 1e-3:
                    break
        # 若两次迭代的omega的差别小于1e-3则终止外层迭代
        diffomega = np.array(list(map(lambda x: abs(x) < 1e-3, pre_omega - omega)))
        if diffomega.all() < 1e-3:
            break'''
    '''#参数设置
    ld = 0.0000000000000001
    beta0 = [1,1,1,1,1,1]
    beta = beta0.copy()
    X, y = read_data('D:/myfile/data/shenjingwangluo/linear-regression-162645/data/exp02/')
    n,p = X.shape
    iter = 0
    diff = 1
    VAL = 10000
    while iter < 100000000000 and diff > 0.0001:
        for j in range(p):
            beta[j] = 0
            y2 = y - X.dot(beta)
            t = X[:,j].dot(y2)/(X[:,j]**2).sum()
            beta[j] = sfun(t,n*ld/(X[:,j]**2).sum())
        VAL2 = np.sum((y-X.dot(beta))**2)+n*ld*np.sum(np.abs(beta))
        diff = np.abs(VAL2 - VAL)
        VAL = VAL2
        iter = iter + 1
    return beta @ data'''
    """
           使用梯度下降法实现Lasso回归

           参数：
           X：训练集特征，形状为（样本数，特征数）
           y：训练集标签，形状为（样本数，1）
           alpha：L1正则化参数
           learning_rate：学习率
           num_iters：迭代次数

           返回值：
           w：Lasso回归模型的参数，形状为（特征数，1）
           """
    #读入数据
    X, y = read_data()
    # 最大最小值归一化
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_normalized = (X - X_min) / (X_max - X_min)
    #X = X_normalized
    # Z-score归一化
    #X_mean = X.mean(axis=1, keepdims=True)
    #X_std = X.std(axis=1, keepdims=True)
    #X_normalized = (X - X_mean) / X_std
    #参数设置
    #初始化参数
    learning_rate = 1e-12
    max_iter = 2500000
    alpha = 0.1
    m, n = X.shape
    w = np.zeros(n)
    b = 2

    # 迭代更新参数
    for i in range(max_iter):
        # 计算梯度
        dw = 1 / m * (X.T @ (X @ w + b - y)) + alpha * np.sign(w)
        db = 1 / m * np.sum(X @ w + b - y)

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 计算损失函数
        J = 1 / (2 * m) * np.sum((X @ w + b - y) ** 2) + alpha * np.sum(np.abs(w))

    return w @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    #print(x)
    y = np.load(path + 'y_train.npy')
    #print(y)
    return x, y

def main():
    data = np.array([1, 2, 3, 4, 5, 6])  # 输入数据
    print(ridge(data))  # 调用main函数，打印一个输出值
    print(lasso(data))

#main()
#D:/myfile/data/shenjingwangluo/linear-regression-162645/data/exp02/
