# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，输出5个类别
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数包含了softmax操作

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x


def build_sample():
    """生成5维向量和最大值索引标签（处理平局情况）"""
    x = np.random.random(5) #随机生成五维向量
    max_val = np.max(x)  #计算最大值
    max_indices = np.where(x == max_val)[0] #找出虽有最大值索引
    y = np.random.choice(max_indices)  # 随机选择一个最大索引
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    """构建包含5个类别的数据集"""
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    """评估模型准确率"""
    model.eval()
    test_sample_num = 100
    X, Y = build_dataset(test_sample_num)
    with torch.no_grad():
        outputs = model(X)
        i, predicted = torch.max(outputs, 1)
        correct = (predicted == Y).sum().item()
    acc = correct / test_sample_num
    print(f"正确预测：{correct}/{test_sample_num}，正确率：{acc:.4f}")
    return acc


def main():
    # 配置参数
    epoch_num = 50  #根据数据量大小选择轮数50
    batch_size = 20  #每轮训练样本20个
    train_sample = 5000 # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 建立初始化模型和优化器
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 加载数据集
    train_x, train_y = build_dataset(train_sample)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = []

        # 数据打乱
        permutation = torch.randperm(train_x.size(0))
        train_x = train_x[permutation]
        train_y = train_y[permutation]

        # 分批次训练
        for i in range(0, train_x.size(0), batch_size):
            batch_x = train_x[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]

            loss = model(batch_x, batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())

        # 记录日志
        avg_loss = np.mean(epoch_loss)
        acc = evaluate(model)
        log.append([acc, avg_loss])
        if acc >= 0.99 :
            break
        print(f"Epoch {epoch + 1}/{epoch_num}，平均loss：{avg_loss:.4f}，验证准确率：{acc:.4f}")

    # 保存模型和可视化
    torch.save(model.state_dict(), "max_classifier.pth")
    plt.plot([l[0] for l in log], label="Accuracy")
    plt.plot([l[1] for l in log], label="Loss")
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    """预测函数"""
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(input_vec))
        probs = torch.softmax(outputs, dim=1)
        _, classes = torch.max(outputs, 1)

    for vec, cls, prob in zip(input_vec, classes, probs):
        print(f"输入：{vec}，预测类别：{cls}，各类别概率：{prob.numpy().round(4)}")


if __name__ == "__main__":
    main()
    # 测试预测
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("max_classifier.pth", test_vec)
