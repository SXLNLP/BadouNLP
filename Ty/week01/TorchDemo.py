# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_size)  # 线性层
                        ,nn.Linear(hidden_size, class_num, bias=False))
        # self.activation = torch.softmax  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x, 1)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return np.argmax(y_pred, 1), np.max(torch.softmax(y_pred, 1).numpy(),axis=1)  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
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
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    #print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        loss = model(x, y)
        '''y_pred = model(x)  # 模型预测
        # print(y_pred)
        # print(y)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 样本判断正确
            else
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))'''
    print("evaluate平均loss：%f" % (loss))
    return loss
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 100  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    hidden_size = 10
    class_num = 5
    learning_rate = 0.003  # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    print(train_y)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        # print(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="loss-eva")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss-tra")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    hidden_size = 10
    class_num = 5
    model = TorchModel(input_size, hidden_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result, prob = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res, prob in zip(input_vec, result, prob):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, res, prob))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pt", test_vec)

