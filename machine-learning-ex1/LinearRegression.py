import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    # 在上面的h(x)函数中是 theta.T*X ,但是在此处却变成了 X*theta.T
    # 因为X是 n*1 纬的向量，在上述公式中的theta是 n*1 纬的向量（参见视频4-1 8分2秒处），视频中要得到一个 1*1 的向量，所以是theta.T*X
    # 而此处我们要得到 n*1 的向量和y做运算， 所以theta是 1*n 的向量（可以参见后面的代码）
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])    # 获得参数theta的总个数，ravel()为扁平化函数
    history_cost = np.zeros(iters)    # 记录每一次迭代产生的代价值

    # iters表示迭代次数
    for i in range(iters):
        error = (X * theta.T) - y    # 表示预测值与实际值之间的误差，对应上式中的 h(x) - y

        # 遍历每个θ
        for j in range(parameters):
            term = np.multiply(error, X[: ,j])
            temp[0 ,j] = theta[0 ,j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        history_cost[i] = computeCost(X, y, theta)

    return theta, history_cost



alpha = 0.01
iters = 1000

data = pd.read_csv('ex1/ex1data1.txt', names=['population', 'profit'])  # 加载数据
data.insert(0, 'ones', 1)

# 设置训练集a
X = data.loc[:, ['ones', 'population']]  # X表示输入变量
y = data.loc[:, ['profit']]  # 表示目标变量

X = np.matrix(X.values)
y = np.matrix(y.values)
# theta = np.matrix(np.array([0,0]))
theta = np.matrix(np.array([0, 0]))

g, history_cost = gradientDescent(X, y, theta, alpha, iters)

x = np.linspace(data.population.min(), data.population.max(), 100)  # 准备绘制直线的x轴数据
f = g[0, 0] + (g[0, 1] * x)  # 通过假设函数h(x)，输入x计算直线y轴的值

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')  # 绘制直线
ax.scatter(data.population, data.profit, label='Traning Data')  # 绘制散点图
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), history_cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
