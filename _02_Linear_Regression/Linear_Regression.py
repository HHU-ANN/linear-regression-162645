# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    #读取数据
    X,y = read_data('D:/myfile/data/shenjingwangluo/linear-regression-162645/data/exp02/')
    #weight = np.ones((6,1))
    for i in range(100):#迭代次数100次
        weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y))
    return weight @ data
    
def lasso(data):
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
def main():
    data = np.array([1, 2, 3, 4, 5, 6])  # 输入数据
    print(ridge(data))  # 调用main函数，打印一个输出值
    print(lasso(data))

#main()