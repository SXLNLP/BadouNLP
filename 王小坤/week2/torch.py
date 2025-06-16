
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

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出5类
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失

    # x: (batch, 5), y: (batch,)
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch, 5)
        if y is not None:
            return self.loss(logits, y.long())  # 交叉熵要求标签为long类型
        else:
            return torch.softmax(logits, dim=1)  # 输出概率


# 生成一个样本，标签为最大值的下标
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签为long类型


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # (batch, 5)
        pred_label = torch.argmax(y_pred, dim=1)
        correct = (pred_label == y).sum().item()
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5
    learning_rate = 0.001
    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        pred_label = torch.argmax(result, dim=1)
    for vec, res, label in zip(input_vec, result, pred_label):
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, int(label), res.numpy()))


if __name__ == "__main__":
    main()
    # test_vec = [
    #     [0.1, 0.2, 0.3, 0.4, 0.9],
    #     [0.9, 0.2, 0.3, 0.4, 0.1],
    #     [0.1, 0.8, 0.3, 0.4, 0.5],
    #     [0.1, 0.2, 0.9, 0.4, 0.5]
    # ]
    # predict("model.bin", test_vec)
