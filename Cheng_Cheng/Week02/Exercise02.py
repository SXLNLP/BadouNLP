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
规律：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


class Multiclassmodel(nn.Module):

    def __init__(self, input_size,num_classes):
        super(Multiclassmodel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用cross entropy损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):

        logits = self.linear(x)  # shape: (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)
        else:
            return torch.softmax(logits, dim=1) # 输出预测结果


# 生成样本，标签为最大值所在的下标（类别 0~4）
def build_sample():
    x = np.random.rand(5)      # 生成一个5维的随机向量（数值范围是0到1之间）
    y = np.argmax(x)           # 找出x中最大值的下标（0到4之间的一个整数）
    return x, y                # 返回输入向量x和标签y


# 构建训练集
def build_dataset(total_num):
    X, Y = [], []
    for _ in range(total_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 评估准确率
def evaluate(model):
    model.eval()
    x, y = build_dataset(100)
    with torch.no_grad():
        y_pred = model(x)  # shape: (batch, 5)
        y_class = torch.argmax(y_pred, dim=1)
        acc = (y_class == y).float().mean().item()
    print(f"准确率: {acc:.4f}")
    return acc


# 主函数
def main():
    input_size = 5
    num_classes = 5
    batch_size = 32
    train_num = 1000
    epoch_num = 40
    lr = 0.01

    model = Multiclassmodel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_x, train_y = build_dataset(train_num)

    log = []
    for epoch in range(epoch_num):
        model.train()
        losses = []
        for i in range(0, train_num, batch_size):
            x = train_x[i:i+batch_size]
            y = train_y[i:i+batch_size]
            loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        acc = evaluate(model)
        print(f"第 {epoch+1} 轮，loss={avg_loss:.4f}")
        log.append([acc, avg_loss])

    torch.save(model.state_dict(), "multi_model.bin")
    print(log)
    plt.plot([x[0] for x in log], label='acc')
    plt.plot([x[1] for x in log], label='loss')
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = Multiclassmodel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_tensor = torch.FloatTensor(input_vec)

    with torch.no_grad():
        logits = model(input_tensor)               # 输出 shape: (batch_size, 5)
        probs = torch.softmax(logits, dim=1)       # 概率分布
        preds = torch.argmax(probs, dim=1)         # 找出最大概率的类别索引

    for vec, prob, pred in zip(input_vec, probs, preds):
        print(f"输入: {vec}, 预测类别: {pred.item()}, 概率分布: {prob.numpy()}")

if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
