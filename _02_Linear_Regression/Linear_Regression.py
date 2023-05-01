# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
    from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
except ImportError as e:
    os.system("sudo pip3 install numpy")
    os.system("sudo pip3 install scikit-learn")
    import numpy as np
    from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV

def ridge(data):
    X,y = read_data()
    
def lasso(data):
    X,y = read_data()
    model = Lasso(alpha=0.00000000000000001)  # 调节alpha可以实现对拟合的程度
    # model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
    # model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
    model.fit(X, y)  # 线性回归建模
    #predicted = model.predict(data)
    print('系数矩阵:\n', model.coef_)
    return model.coef_ @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

def main():
    data = np.array([1, 2, 3, 4, 5, 6])  # 输入数据
    #print(ridge(data))  # 调用main函数，打印一个输出值
    print(lasso(data))

main()
#'D:/myfile/data/shenjingwangluo/linear-regression-162645/data/exp02/'
