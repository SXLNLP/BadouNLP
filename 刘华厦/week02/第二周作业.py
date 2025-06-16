import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，交叉熵实现一个多分类任务，最大的数字在哪维就属于哪一类。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 线性层
        self.linear = nn.Linear(input_size, 5)
        # 激活函数
        self.activation = torch.sigmoid
        # 损失函数
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y):
        x = self.linear(x)
        y_pred = self.activation(x)
        return self.loss(y_pred, y)

    def predict(self, x):
        return self.activation(self.linear(x))


# 生成随机样本
def build_example():
    x = np.random.random(5)
    # # 最大值所在维度
    # y = [0.1, 0.1, 0.1, 0.1, 0.1]
    # x_max_index = np.argmax(x)
    # y[x_max_index] = 0.6
    x = torch.FloatTensor(x)
    # 归1化
    y = torch.softmax(x, dim=0)
    return x.numpy(), y.numpy()


# 生成一批数据
def build_dataset(data_num):
    data_x = []
    data_y = []
    for i in range(data_num):
        x, y = build_example()
        data_x.append(x)
        data_y.append(y)

    return torch.FloatTensor(data_x), torch.FloatTensor(data_y)


# 测试模型准确率
# 返回准确率
def evaluate(torchModel):
    torchModel.eval()
    # 生成测试数据
    data_num = 100
    data_x, data_y = build_dataset(data_num)
    # 预测正确的数量和错误的数量
    correct_num = 0
    wrong_num = 0
    with torch.no_grad():
        y_pred = torchModel.predict(data_x)
        for y_p, y_t in zip(y_pred, data_y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct_num += 1
            else:
                wrong_num += 1
    return correct_num / (correct_num + wrong_num)


def main():
    # 轮次
    epoch_num = 500
    # 每轮样本数
    batch_size = 100
    # 训练数据
    data_num = 2000
    data_x, data_y = build_dataset(data_num)
    # 输入向量维度
    input_size = 5
    # 学习率
    learning_rate = 0.01
    # 建立模型
    torchModel = TorchModel(input_size)
    # 优化器
    optimizer = torch.optim.Adam(torchModel.parameters(), lr=learning_rate)
    # 收集准确率和损失值
    log = []
    for epoch in range(epoch_num):
        # 使用训练模式
        torchModel.train()
        # 收集损失值
        watch_loss = []
        for batch in range(data_num // batch_size):
            train_x = data_x[batch * batch_size: (batch + 1) * batch_size]
            train_y = data_y[batch * batch_size: (batch + 1) * batch_size]
            # 计算损失
            loss = torchModel.forward(train_x, train_y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optimizer.step()
            # 梯度归零
            optimizer.zero_grad()

            # 查看损失值
            print(f"损失值loss = {loss}")
            watch_loss.append(loss.item())

        # 测试准确率
        correct_rate = evaluate(torchModel)
        print(f"第{epoch}轮，准确率={correct_rate}")
        # 收集准确率和损失值
        log.append([correct_rate, float(np.mean(watch_loss))])

        # if correct_rate > 0.9999:
        #     break

    # 保存模型（训练好的权重）
    torch.save(torchModel.state_dict(), './myWorkModel.pth')

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="correct_rate")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
