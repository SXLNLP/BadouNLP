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
规律：使用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类

"""


class Classifier(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.layer = nn.Linear(input_size, 5)  # 输入输出均为5维

    def forward(self, x):
        return self.layer(x)  # 输出logits


# 损失函数
ce_loss = nn.CrossEntropyLoss()


# 随机生成一批样本
def build_dataset(total_sample_num):
    inputs = torch.randn(total_sample_num, 5)  # 输入：4个样本，每个5维
    labels = torch.argmax(inputs, dim=1)  # 标签：每个样本最大值的位置
    return inputs, labels


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.1  # 学习率
    # 建立模型
    model = Classifier(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    inputs, labels = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        outputs = model(inputs)
        watch_loss = []
        loss = ce_loss(outputs, labels)  # 计算loss
        loss.backward()  # 计算梯度
        optim.step()  # 更新权重
        optim.zero_grad()  # 梯度归零
        watch_loss.append(loss.item())
        print("=========\n第%d轮 平均loss: %f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 测试预测
    model.eval()  # 测试模式
    with torch.no_grad():
        test_input = torch.randn(5)
        prediction = torch.argmax(model(test_input))
        print(f"\nTest input: {test_input.numpy().round(2)}")
        print(f"\n训练预测: {prediction}")
        print(f"\n所属分类: {torch.argmax(test_input).item()}")


if __name__ == "__main__":
    main()
