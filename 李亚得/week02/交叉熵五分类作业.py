# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

1、基于pytorch框架编写模型训练
2、实现一个自行构造的找规律(机器学习)任务
3、规律：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()

        # 第一件事：定义要使用哪些网络层
        # 线性层1，Linear(输入纬度，输出纬度)，输入输出纬度决定A的形状
        self.linear = nn.Linear(input_size, num_classes)
        # 多分布使用softmax做归一化，nn.Softmax, 但交叉熵CrossEntropyLoss会默认做一次softmax
        self.activation = nn.Softmax(dim=1)

        # loss函数采用交叉熵
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失



    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 第二件事：要定义这些网络层是如何组合的
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        #y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            lossvalue = self.loss(x, y)
            return lossvalue  # 预测值和真实值计算损失
        else:
            return torch.argmax(x, dim=1)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大数字落在哪一纬，就是哪一类
def build_sample():
    x = np.random.random(5)
    # argmax方法用户获取最大元素索引
    # 返回值：[随机五维向量，五维向量的最大值下标]
    return x, np.argmax(x)



# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个0，%d个1，%d个2，%d个3，%d个4" % (sum(y==0),sum(y==1),sum(y==2),sum(y==3),sum(y==4)))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100 # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5   # 输出为一个数，代表属于哪一类（用下标表示）
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  等价于model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 优化器.更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    print("model loading...")
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    print("complete!")
    model.eval()  # 测试模式

    print("let's play : ")
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("%s, %d" % (vec, res))


if __name__ == "__main__":
    main()
    #test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #            [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #            [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #            [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    #predict("model.bin", test_vec)
    test_vec = np.random.randn(10, 5)
    predict("model.bin", test_vec)
