# coding:utf8
# zhouqp-周青平
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，返回最大值所在维度
"""
class FiveDimClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)  # 五维输入输出
        self.softmax = nn.Softmax(dim=1)  # 沿特征维度归一化

    def forward(self, x,):
        return self.linear(x)


def build_sample(batch_size=20):
    # 生成随机五维向量，最大值索引作为标签
    x = torch.randn(batch_size, 5)  # 形状[batch_size, 5]
    y = torch.argmax(x, dim=1)  # 取每行最大值索引
    return x, y




def main():
    # 配置参数
    epoch_num = 500  # 训练轮数
    batch_size = 100  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    learning_rate = 0.1  # 学习率

    # log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_sample(train_sample)

    # 建立模型
    model = FiveDimClassifier()
    ce_loss = nn.CrossEntropyLoss()  # 内置softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            outputs = model(x)
            loss = ce_loss(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            watch_loss.append(loss.item())
            avg_loss=np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        if avg_loss<0.01:
            break
    # 保存模型
    torch.save(model.state_dict(), "model_five.bin")
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = FiveDimClassifier()
    model.load_state_dict(torch.load(model_path,map_location="cpu", weights_only=True))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = torch.argmax(model(torch.tensor(input_vec)), dim=1)
        print("result=",result)
    # for vec, res in zip(input_vec, result):
    #     print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果




if __name__ == "__main__":
    main()
    # testx = [[0.07889086,0.10229675,1.31082123,0.03504317,0.88920843],
    #             [0.74963533,3.5524256,2.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # print(testx)
    # print(torch.argmax(torch.tensor(testx), dim=1))
    # predict("model_five.bin", testx)
