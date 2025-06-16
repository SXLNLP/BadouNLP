# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5个维度
        self.loss = nn.CrossEntropyLoss()  # 改用交叉熵损失

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 直接输出logits
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 处理标签形状
        else:
            return torch.softmax(y_pred, dim=1)  # 预测时返回概率分布


# 生成样本（改为五分类）
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)  # 返回最大值所在维度索引


# 生成数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签改为LongTensor


# 评估函数修改
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计各类别数量
    unique, counts = np.unique(y.numpy(), return_counts=True)
    print("各类别数量：", dict(zip(unique, counts)))

    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y.squeeze()).sum().item()

    acc = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{acc:.4f}")
    return acc


def main():
    # 参数保持不变
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch_idx in range(train_sample // batch_size):
            # 获取批次数据
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            x = train_x[start:end]
            y = train_y[start:end]

            # 计算损失
            loss = model(x, y)
            total_loss += loss.item()

            # 优化步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / (train_sample // batch_size)
        print(f"Epoch {epoch + 1}/{epoch_num}, Loss: {avg_loss:.4f}")

        acc = evaluate(model)
        log.append([acc, avg_loss])

    # 保存和可视化
    torch.save(model.state_dict(), "multiclass_model.pth")
    plt.plot([x[0] for x in log], label="Accuracy")
    plt.plot([x[1] for x in log], label="Loss")
    plt.legend()
    plt.show()


# 预测函数修改
def predict(model_path, input_vec):
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        probabilities = model(torch.FloatTensor(input_vec))
        predictions = torch.argmax(probabilities, dim=1)

    for vec, prob, pred in zip(input_vec, probabilities, predictions):
        print(f"输入：{vec}")
        print(f"各类别概率：{prob.numpy().round(4)}")
        print(f"预测类别：{pred.item()}\n")


if __name__ == "__main__":
    main()

    # 测试样例
    test_vectors = [
        [0.9, 0.1, 0.2, 0.3, 0.4],  # 第0类
        [0.1, 0.95, 0.2, 0.3, 0.4],  # 第1类
        [0.1, 0.2, 0.98, 0.3, 0.4],  # 第2类
        [0.1, 0.2, 0.3, 0.99, 0.4],  # 第3类
        [0.1, 0.2, 0.3, 0.4, 0.97]  # 第4类
    ]
    predict("multiclass_model.pth", test_vectors)
