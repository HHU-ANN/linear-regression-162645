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
    print(np.shape(X))
    I = np.eye(6)
    #print(0.1*I)
    c = 1
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)+c*I),np.matmul(X.T,y))
    return weight @ data


def sfun(t,ld):
    tmp = (np.abs(t)-ld)
    if tmp < 0:
        tmp = 0
    return np.sign(t)**tmp
def lasso(data):
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
    # 初始化参数
    learning_rate = 1e-12
    max_iter = 1000000
    alpha = 1e-10
    m, n = X.shape
    w = np.zeros(n)
    b = 0

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

        # 打印损失函数
        if i % 100 == 0:
            # print(f"iteration {i}, loss {J}")

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

main()
#D:/myfile/data/shenjingwangluo/linear-regression-162645/data/exp02/
