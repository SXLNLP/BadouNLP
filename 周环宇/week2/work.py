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
规律：x是一个5维向量，1.如果向量中的5个值均不相等，则值最大的那个置为1，其余置为0; 2.如果存在相等的值，则第一位置为1，其余置为0


存在问题：
1.  对于逻辑2预测不准，哪怕是在某次损失函数正确率为100%时，对于最终的训练模型测试依旧预测失败，
原因可能在于在生成训练数据的时候，就没有生成逻辑2情况的数据，但是训练结果不符合实际逻辑。

2.  后续添加 build_sample_with_duplicate 函数用于强制生成适用的数据，但是导致无论拉高多少训练轮次，训练结果的的准确率都过低，
故而此模型可能并不适合用于我预定情况的训练；但是，如果不添加逻辑2，则可以较好的拟合函数。
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.linear(x)  # 线性计算

        if y is not None:
            # 对于交叉熵损失，需要将y转换为类别索引
            y_class = torch.argmax(y, dim=1)
            return self.loss(x, y_class)  # 计算损失
        else:
            return torch.softmax(x, dim=1)  # 预测时手动添加Softmax


# 生成一个样本
def build_sample():
    x = np.random.random(5)  # 生成5维随机向量

    # 检查所有值是否唯一
    if len(np.unique(x)) == 5:  # 所有值都不相等
        label = np.zeros(5)  # 创建全0标签
        label[np.argmax(x)] = 1  # 将最大值位置设为1
    else:  # 存在相等的值
        label = np.zeros(5)
        label[0] = 1  # 第一位置为1

    return x, label  # 返回特征和标签


# 随机生成一批样本
# def build_dataset(total_sample_num):
#     X = []
#     Y = []
#     for i in range(total_sample_num):
#         x, y = build_sample()
#         X.append(x)
#         Y.append(y)
#     return torch.FloatTensor(X), torch.FloatTensor(Y)


# 生成一个强制包含重复值的样本
def build_sample_with_duplicate():
    x_part = np.random.random(4)  # 生成前4个随机数
    duplicate_value = np.random.choice(x_part)  # 随机选择一个值作为重复值
    x = np.append(x_part, duplicate_value)  # 添加重复值构成5维向量
    np.random.shuffle(x)  # 打乱顺序
    # 存在重复值时，第一位置1
    label = np.zeros(5)
    label[0] = 1
    return x, label


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        # 当总样本数>5且当前是第5、10、15...个样本时，强制生成重复值样本
        # if total_sample_num > 5 and (i + 1) % 5 == 0:
        #     x, y = build_sample_with_duplicate()
        # else:
        #     x, y = build_sample()  # 其他情况正常生成
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # 比较预测类别和真实类别
        predicted_classes = torch.argmax(y_pred, dim=1)
        true_classes = torch.argmax(y, dim=1)
        correct = (predicted_classes == true_classes).sum().item()

    accuracy = correct / test_sample_num
    print("本次预测个数：%d,正确预测个数：%d, 正确率：%f" % (test_sample_num, correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.001  # 学习率
    input_size = 5  # 输入维度

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        if acc > 0.9999:
            break

    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 画图
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

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, predicted_class, res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [
        [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 最大值在最后
        [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],  # 最大值在中间
        [0.80797868, 0.63482528, 0.67482528, 0.34675372, 0.19871392],  # 最大值在最前
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.92579291]  # 有重复值
    ]
    predict("model.bin", test_vec)
