import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律（回归）任务

规律：x 是一个 10 维向量，标签 y 是由 x 中各特征经过非线性组合构造出的复杂函数，
包括 sin、log、平方、乘积、tanh 等非线性表达式，并加入高斯噪声

目标：训练一个神经网络回归模型，学习从输入向量 x 预测连续实数标签 y 的规律

损失函数：均方误差损失函数（MSELoss）
评估方式：输出在测试集上的均方误差（MSE）和平均绝对误差（MAE）
"""


def generate_data(n_samples, noise_std=0.2, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-3, 3, size=(n_samples, 10))  # 10个feature，每个在 [-3, 3]

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X.T  # 解包变量

    # 构造复杂目标函数
    y = (
            0.8 * np.sin(x1) +
            0.5 * np.log(np.abs(x2) + 1) +
            0.3 * x3 ** 2 -
            0.4 * x4 * x5 +
            0.6 * np.exp(-x6) +
            0.2 * x7 * x8 +
            0.5 * np.tanh(x9 + x10)
    )

    # 加噪声
    noise = np.random.normal(0, noise_std, size=n_samples)
    y += noise

    return X, y


class TorchModel(nn.Module):  # 既包含了训练，也包含预测，具体看有没有传 y。实现自动化
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, out_features=input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, 1)
        self.loss = nn.MSELoss()

    def forward(self, x, y=None):
        x = self.fc1(x)
        x = self.relu(x)
        y_predict = self.fc2(x)
        if y is not None:  # 调用函数传 y 了，则 y 不为 None 。此时为训练
            return self.loss(y_predict, y)
        else:  # 调用函数没传 y，则 y 为默认的 None 。此时为预测
            return y_predict


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x_test, y_test = generate_data(test_sample_num)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)  # 确保形状一致

    with torch.no_grad():
        y_pred = model(x_test)  # (100, 1)
        loss_fn = torch.nn.MSELoss()
        total_loss = loss_fn(y_pred, y_test).item()

        # 计算平均绝对误差（MAE）和平均相对误差
        total_absolute_error = torch.mean(torch.abs(y_pred - y_test)).item()

    print("评估样本数：%d" % test_sample_num)
    print("均方误差（MSE）：%.4f" % total_loss)
    print("平均绝对误差（MAE）：%.4f" % total_absolute_error)

    return total_loss, total_absolute_error


def main():
    epoch_num = 100
    batch_size = 20
    train_sample = 5000
    input_size = 10
    learning_rate = 0.001

    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = generate_data(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = torch.FloatTensor(train_x[batch_index * batch_size: (batch_index + 1) * batch_size])
            y = torch.FloatTensor(train_y[batch_index * batch_size: (batch_index + 1) * batch_size]).view(-1, 1)

            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        total_loss, total_absolute_error = evaluate(model)  # 测试本轮模型结果
        log.append([total_absolute_error, float(np.mean(watch_loss))])
    print("=========\n第%d轮平均loss:%.6f" % (epoch_num, log[-1][1]))
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="MAE")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    main()
