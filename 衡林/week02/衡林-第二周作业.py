import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大的数字在哪维就属于哪一类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    return x, x.argmax()


def build_dataset(total_sample_num):
    samples = [build_sample() for _ in range(total_sample_num)]
    X, Y = zip(*samples)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y).sum().item()
    accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5  # 输入维度
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建数据集
    train_x, train_y = build_dataset(train_sample)
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in dataloader:
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model)
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "model1.pt")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
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
        print("输入：%s, 概率值：%s, 预测类别：%d" % (vec, res, int(torch.argmax(res))))


if __name__ == "__main__":
    main()
    test_vec = [
        [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
        [0.74963533, 0.5524256, 0.95758807, 0.98820434, 0.44890681],
        [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.19349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]
    ]
    predict("model1.pt", test_vec)
