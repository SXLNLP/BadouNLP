# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

修改说明：
1. 将二分类任务改为五分类任务
2. 新规律：x是一个5维向量，标签由5个维度中的最大值索引决定（0-4类）
3. 使用交叉熵损失函数

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5个神经元，对应5类
        self.loss = nn.CrossEntropyLoss()  # 改为交叉熵损失函数

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 输入需要是logits，y是类别索引（非one-hot）
        else:
            return x  # 直接返回logits

# 生成一个样本（五分类逻辑）
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)  # 新规律：最大值索引作为类别（0-4）
    return x, label

# 生成数据集（标签改为整数）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 直接存储整数标签
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意标签类型为Long

# 修改评估函数适应五分类
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计各类别数量
    class_count = [0] * 5
    for label in y:
        class_count[label] += 1
    print("各类别样本数量：", class_count)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
        correct = (predicted == y).sum().item()
        wrong = test_sample_num - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    # 初始化模型
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 加载数据集
    train_x, train_y = build_dataset(train_sample)

    # 训练循环（保持不变结构）
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 获取批次数据
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]

            # 计算损失
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    return

# 修改预测函数显示概率分布
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        probabilities = torch.softmax(logits, dim=1)  # 转换为概率分布

    for vec, prob in zip(input_vec, probabilities):
        print(f"\n输入：{vec}")
        print("预测类别分布：")
        pred_class = torch.argmax(prob).item()
        for i, p in enumerate(prob):
            print(f"  类别{i}: {p*100:.2f}%")
        print(f"最大概率类别：{pred_class}")
if __name__ == "__main__":
    main()

    # 测试样例
    test_vec = [
        [0.9, 0.1, 0.1, 0.1, 0.1],  # 应预测类别0
        [0.1, 0.8, 0.1, 0.1, 0.1],  # 应预测类别1
        [0.1, 0.1, 0.7, 0.1, 0.1],  # 应预测类别2
        [0.1, 0.1, 0.1, 0.6, 0.1],  # 应预测类别3
        [0.1, 0.1, 0.1, 0.1, 0.95]  # 应预测类别4
    ]
    predict("model.bin", test_vec)
