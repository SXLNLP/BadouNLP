# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
改用交叉熵实现一个5分类任务，五维随机向量最大的数字在哪维就属于哪一类
"""
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5个神经元对应5个类别
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 索引为整数，需要long类型标签
        else:
            return y_pred  # 输出预测结果

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，取最大值所在的索引作为类别
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 取最大值所在的索引作为类别（0-4）
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 索引为整数 注意标签类型为LongTensor


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计各类别数量
    class_count = [0] * 5
    for label in y:
        class_count[label] += 1
    print("各类别数量:", class_count)

    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y.squeeze()).sum().item()
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

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

        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss", color='red')
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(input_vec))
        probabilities = torch.softmax(y_pred, dim=1)  # 转换为概率

    for vec, prob, pred in zip(input_vec, probabilities, y_pred):
        class_idx = torch.argmax(pred).item()
        print(f"输入：{vec}")
        print(f"预测类别：{class_idx}")
        print(f"类别概率：{prob.numpy().round(4)}")
        print("=" * 30)
        # 打印结果


if __name__ == "__main__":
    # main()

    # 测试预测
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
