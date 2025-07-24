# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch.jit._shape_functions import max_dim

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果哪个数大，则输出几维

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.softmax
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 直接输出logits
        if y is not None:
            # CrossEntropyLoss需要y的形状为(batch_size,)，且类型为long
            return self.loss(y_pred, y.squeeze().long())
        else:
            return self.activation(y_pred, dim=1)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量,找出最大值的维度
def build_sample():
    x = np.random.random(5)  # 生成5维随机向量
    max_dim = np.argmax(x)  # 找出最大值的维度（0~4）
    return x, max_dim

# 随机生成一批样本

def build_dataset(total_sample_num):
    # 预分配 NumPy 数组（比列表追加更快）
    X = np.zeros((total_sample_num, 5))  # (n_samples, 5)
    Y = np.zeros((total_sample_num, 1))  # (n_samples, 1)

    for i in range(total_sample_num):
        x, y = build_sample()
        X[i] = x  # 直接赋值，避免列表追加
        Y[i] = y

    # 一次性转为 Tensor（避免逐元素转换）
    return torch.FloatTensor(X), torch.LongTensor(Y)  # Y 是类别标签，用 LongTensor

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计各类别数量
    unique, counts = torch.unique(y, return_counts=True)
    count_dict = dict(zip(unique.tolist(), counts.tolist()))
    print("本次预测集中各类别数量:", {k + 1: count_dict.get(k, 0) for k in range(5)})

    correct = 0
    with torch.no_grad():
        y_prob = model(x)  # 获取概率分布
        y_pred = torch.argmax(y_prob, dim=1)  # 获取预测类别
        correct = (y_pred == y).sum().item()

    accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
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
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
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
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec)
        probabilities = model(input_tensor)
        predictions = torch.argmax(probabilities, dim=1)

        for vec, prob, pred in zip(input_vec, probabilities, predictions):
            print(f"输入：{vec}, 预测类别：{pred.item() + 1}, 各类别概率：{prob.tolist()}")


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                 [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                 [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                 [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
