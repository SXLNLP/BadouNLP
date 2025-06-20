"""
该例子用python基础代码实现了最基础的深度学习原理案例，根据X预测Y
深度学习基本流程：
1、找样本数据，或者自己生成
2、选模型（其实就是选个函数）
3、定义损失函数，例如（预测值-真实值）**2
4、模型参数初始化
5、利用优化器，使用梯度下降算法（w_new = w_old - rate * gradient），迭代更新模型参数。w是模型参数，rate是学习率，gradient是损失函数对w求偏导
"""

import matplotlib.pyplot as pyplot

"""
****************注意：示例代码中每一行代码都要搞懂****************
"""

# 1、生成样本数据   
X = [0.01 * x for x in range(200)]
Y = [(3*x**2 + 5*x + 8) for x in X]
print("X：\n", X)
print("Y：\n", Y)

#2、选模型（其实就是选个函数）
def func(x):
    y = w1 * x **2 + w2 * x + w3
    return y

#3、定义损失函数（其实就是选个函数）
def loss(y_pred, y_true):
    y = (y_pred - y_true)**2
    return y

#4、模型参数初始化(一般初始值设置为-1~1之间，相差不要太大)
w1 = 1
w2 = 0.5
w3 = 1

#学习率
lr = 0.01

#一共训练几轮
epoch_size = 1000
#每一轮训练的时候，几个样本计算loss的平均值，以便尽可能消除个别样本对整体造成的影响
batch_size = 10

for epoch in range(epoch_size):
    grad_w1 = 0
    grad_w2 = 0
    grad_w3 = 0
    loss_epoch = 0
    counter = 0

    for x, y_true in zip(X, Y):   ####zip函数用来将X、Y这两个列表组装成一个可迭代对象，即一一对应起来。否则就得遍历一个集合，另外一个用下标读取元素
        y_pred = func(x)
        loss_epoch += loss(y_pred, y_true)
        counter +=1

        grad_w1 += 2 * (y_pred - y_true) * x **2
        grad_w2 += 2 * (y_pred - y_true) * x
        grad_w3 += 2 * (y_pred - y_true)

        if counter == batch_size:
            #更新权重
            w1 = w1 - lr * grad_w1 / batch_size
            w2 = w2 - lr * grad_w2 / batch_size
            w3 = w3 - lr * grad_w3 / batch_size

            counter = 0
            grad_w1 = 0
            grad_w2 = 0
            grad_w3 = 0

    loss_epoch = loss_epoch / len(X)

    print("第%d轮，loss：%f"%(epoch, loss_epoch))

    if loss_epoch<0.000001:
        break

print(f"训练后的模型参数 w1：{w1}，w2：{w2}，w3：{w3}")

#使用训练后模型输出预测值
Yp = [func(i) for i in X]

#预测值与真实值比对数据分布
pyplot.scatter(X, Y, color="red")
pyplot.scatter(X, Yp)
pyplot.show()
