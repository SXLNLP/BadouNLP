import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于PyTorch实现5分类任务：
规律：5维向量x的标签为其最大值的索引（0-4）
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出维度为类别数（5）
        self.activation = nn.Softmax(dim=1)  # 多分类使用Softmax，在维度1上计算
        self.loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数（内置Softmax）

    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss_func(x, y.long())  # 交叉熵损失要求标签为LongTensor
        else:
            return self.activation(x)  # 输出概率分布


# 生成单个样本：标签为5维向量中最大值的索引（0-4）
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)  # 最大值的索引作为标签（0-4）
    return x, label


# 生成批量数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 标签为0-4的整数
    # 转为张量
    X = torch.FloatTensor(np.array(X))
    Y = torch.LongTensor(np.array(Y))  # 交叉熵损失要求标签为Long类型
    return X, Y


# 评估模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print(f"测试集标签分布：{np.bincount(y.numpy())}")  # 统计各标签数量

    correct = 0
    with torch.no_grad():
        y_pred = torch.argmax(model(x), dim=1)  # 取概率最大的索引作为预测结果
        correct = (y_pred == y).sum().item()

    accuracy = correct / test_sample_num
    print(f"准确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 30
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5  # 5分类
    learning_rate = 0.01

    # 初始化模型与优化器
    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 生成训练数据
    train_x, train_y = build_dataset(train_sample)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch_idx in range(train_sample // batch_size):
            start = batch_idx * batch_size
            end = start + batch_size
            x_batch = train_x[start:end]
            y_batch = train_y[start:end]

            loss = model(x_batch, y_batch)  # 计算交叉熵损失
            loss.backward()
            optim.step()
            optim.zero_grad()

            total_loss += loss.item()

        # 计算平均损失并评估
        avg_loss = total_loss / (train_sample // batch_size)
        acc = evaluate(model)
        log.append([acc, avg_loss])
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "model_5class.bin")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(epoch_num), [l[0] for l in log], label="Accuracy", marker='o')
    plt.plot(range(epoch_num), [l[1] for l in log], label="Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# 预测函数
def predict(model_path, input_vecs):
    model = TorchModel(5, 5)  # 输入维度5，输出类别5
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(np.array(input_vecs))
        y_pred = torch.argmax(model(x), dim=1)  # 预测类别索引
        probs = model(x).numpy()  # 概率分布

    for vec, label, prob in zip(input_vecs, y_pred, probs):
        print(f"输入：{vec:.4f}, 预测类别：{label.item()}, 概率分布：{prob}")


if __name__ == "__main__":
    main()

    # 测试预测（示例输入：最大值位置分别为0,1,2,3,4）
    test_vecs = [
        [0.9, 0.1, 0.1, 0.1, 0.1],  # 最大值索引0
        [0.1, 0.8, 0.1, 0.1, 0.1],  # 最大值索引1
        [0.1, 0.1, 0.7, 0.1, 0.1],  # 最大值索引2
        [0.1, 0.1, 0.1, 0.6, 0.1],  # 最大值索引3
        [0.1, 0.1, 0.1, 0.1, 0.5]  # 最大值索引4
    ]
    predict("model_5class.bin", test_vecs)
