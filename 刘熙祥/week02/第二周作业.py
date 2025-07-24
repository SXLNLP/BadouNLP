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

改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size=10):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)  # 线性层
        #self.hidden = nn.Linear(input_size, hidden_size)  # 隐藏层
        #self.activation = nn.ReLU()
        #self.activation = nn.GELU()
        #self.activation = torch.softmax  # 多分类适合用softmax 但是损失函数中已经有了softmax，二分类适合用sigmoid
        #self.output = nn.Linear(hidden_size, 5)
        self.ce_loss = nn.CrossEntropyLoss()  # loss函数改为交叉熵Cross Entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        #x = self.hidden(x)  # (batch_size, input_size) -> (batch_size, hidden_size)
        #x = self.activation(x)  # (batch_size, 5) -> (batch_size, 5)
        #y_pred = self.output(x)  # (batch_size, hidden_size) -> (batch_size, 5)
        if y is not None:
            return self.ce_loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.argmax(x, dim=1)  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，五维随机向量最大的数字在哪维就属于哪一类
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    if max_index == 0:
        return x, 0
    elif max_index == 1:
        return x, 1
    elif max_index == 2:
        return x, 2
    elif max_index == 3:
        return x, 3
    elif max_index == 4:
        return x, 4


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
    print("本次预测集中共有%d个样本" % sum(y))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        # if y_pred == y:
        #     correct += 1  # 预测正确的样本数
        # else:
        #     wrong += 1
        correct += (y_pred == y).sum().item() #形状相同不用if可以直接比对，用sum()求和，用item()转化为整型数字
        # for y_p, y_t in zip(y_pred, y):  #无法用if直接比对，必须用循环逐个比对
        #     print(y_p, y_t)
        #     if y_p == y_t:# 与真实标签进行对比
        #         correct += 1  # 预测正确的样本数
        #     else:
        #         wrong += 1
        wrong = test_sample_num - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率需要调整，提示准确率
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

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    #main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
