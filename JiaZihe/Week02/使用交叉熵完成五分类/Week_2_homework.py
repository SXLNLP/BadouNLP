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
规律：x是一个5维向量，输出最大数值的位置

"""


class Week2_HW_Model(nn.Module):
    def __init__(self, input_size):
        super(Week2_HW_Model, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, y=None):
    #     x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
    #     y_pred = [0, 0, 0, 0, 0]
    #     y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
    #     if y is not None:
    #         return self.loss(y_pred, y)  # 预测值和真实值计算损失
    #     else:
    #         return y_pred  # 输出预测结果
    def forward(self, x, y=None):
        # 直接输出logits，不要加激活函数
        logits = self.linear(x)  # (batch_size, 5)
        pred = torch.argmax(logits, dim=1)

        if y is not None:
            loss = nn.CrossEntropyLoss()(logits, y)
            return loss, pred
        else:
            # 预测时可以返回softmax概率
            return pred


# 生成一个样本
# 随机生成一个5维向量，找到最大值和它的位置，
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

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
    print("本次预测集中共有%d个样本" % (test_sample_num))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.7 :
                wrong += 1  # 负样本判断正确
            else:
                correct += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 建立模型和优化器
    model = Week2_HW_Model(input_size)  # 使用直接输出类别的模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        watch_acc = []  # 新增：记录每batch准确率

        for batch_index in range(train_sample // batch_size):
            # 获取当前batch数据
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            # 前向传播
            loss, pred = model(x, y)  # 注意现在返回两个值

            # 反向传播
            loss.backward()
            optim.step()
            optim.zero_grad()

            # 记录loss和准确率
            watch_loss.append(loss.item())
            batch_acc = (pred == y).float().mean().item()  # 计算当前batch准确率
            watch_acc.append(batch_acc)

        # 打印本轮结果
        avg_loss = np.mean(watch_loss)
        avg_acc = np.mean(watch_acc)  # 计算平均准确率
        print(f"=========\n第{epoch + 1}轮结果: 平均loss={avg_loss:.4f}, 平均准确率={avg_acc:.4f}")
        log.append([avg_acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = Week2_HW_Model(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
