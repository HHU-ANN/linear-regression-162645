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
    c = 0.1
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)+c*I),np.matmul(X.T,y))
    return weight @ data
    
def lasso(data):
    X, y = read_data()
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
    return weight @ data

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