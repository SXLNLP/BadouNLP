# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt



"""

基于pytorch框架编写模型作业
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，改用交叉熵实现一个多分类任务，5维随机向量最大的数字在哪维就属于哪一类。

"""


# 定义神经网络
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.CrossEntropyLoss()  # loss 交叉熵函数

    def forward(self, x, y=None):
        x = self.linear(x)  # 过线性层
        if y is not None:
            return self.loss(x, y.squeeze().long())  # 计算损失
        else:
            return x  # 输出预测结果（logits）


# 设计任务样本
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # numpy中argmax函数，取最大值
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    # 用array将列表合并成一个高维数组，可以提升pytorch处理效率
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 测试样本
# total_sample_num = 10
# x, y = build_dataset(total_sample_num)
# print("训练集合x（10个5维向量）：", x)
# print("验证集合y（10个分类结果）：", y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    X, Y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(X)
        y_pred_labels = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(y_pred_labels, Y.squeeze()):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数,实际项目无法改变，根据数据量来
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

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
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="Acc")  # 画Acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")  # 画Loss曲线
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, torch.argmax(res)))


if __name__ == "__main__":
    main()
    test_vec = [
        [0.21, 0.41, 0.32, 0.44, 0.12],
        [0.54, 0.23, 0.55, 0.22, 0.52],
        [0.32, 0.53, 0.53, 0.43, 0.11],
        [0.13, 0.43, 0.56, 0.43, 0.25]
    ]
    predict("model.bin", test_vec)
