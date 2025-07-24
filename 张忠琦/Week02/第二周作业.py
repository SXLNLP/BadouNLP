import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的多分类找规律任务
规律：x是一个5维向量，最大值所在的索引位置即为类别标签(0-4)
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测的类别分数
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测的类别分数


# 生成一个样本
# 随机生成一个5维向量，最大值所在的索引位置即为类别标签(0-4)
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
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计每个类别的样本数量
    class_counts = [0] * 5
    for label in y:
        class_counts[label.item()] += 1
    print("测试集中各类样本数量分布:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y).sum().item()
        wrong += len(y) - correct
    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong):.4f}")
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数增加，多分类任务通常需要更多训练
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    num_classes = 5  # 分类类别数

    # 建立模型
    model = TorchModel(input_size, num_classes)

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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.6f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model_multi.bin")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss", color='orange')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测，输出类别分数
        _, predicted = torch.max(result, 1)  # 获取预测的类别索引

    for vec, pred, scores in zip(input_vec, predicted, result):

        probs = torch.softmax(scores, dim=0).numpy()
        print(f"输入：{vec}")
        print(f"预测类别：{pred.item()}")
        print(f"概率分布：{[f'{p:.4f}' for p in probs]}")
        print(f"最大值索引：{np.argmax(vec)}, 真实类别：{np.argmax(vec)}\n")


if __name__ == "__main__":
    main()

    # 测试向量
    test_vec = [
        [0.1, 0.8, 0.3, 0.4, 0.5],  # 最大值在索引1
        [0.9, 0.2, 0.3, 0.4, 0.5],  # 最大值在索引0
        [0.1, 0.2, 0.9, 0.4, 0.5],  # 最大值在索引2
        [0.1, 0.2, 0.3, 0.8, 0.5],  # 最大值在索引3
        [0.1, 0.2, 0.3, 0.4, 0.9]  # 最大值在索引4
    ]

    predict("model_multi.bin", test_vec)
