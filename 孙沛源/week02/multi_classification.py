# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 任务：输入一个 5 维向量，输出最大值所在的下标（0-4），作为类别

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出维度改为类别数
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    def forward(self, x, y=None):
        logits = self.linear(x)  # 输出 logits: [batch, num_classes]
        if y is not None:
            return self.loss(logits, y)  # y shape: [batch], 值是 0~4
        else:
            return torch.softmax(logits, dim=1)  # 返回概率分布（可选）

# 生成一个样本：x 是 5维向量，y 是最大值所在的索引
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

# 生成一批训练样本
def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意：标签必须是 Long 类型

# 评估准确率
def evaluate(model):
    model.eval()
    x, y = build_dataset(100)
    with torch.no_grad():
        pred = model(x)  # shape: [100, 5]
        pred_labels = torch.argmax(pred, dim=1)
        correct = (pred_labels == y).sum().item()
        acc = correct / len(y)
        print("预测正确个数: {}, 正确率: {:.2f}".format(correct, acc))
        return acc

# 主训练流程
def main():
    input_size = 5
    num_classes = 5
    epoch_num = 20
    batch_size = 32
    train_sample = 2000
    learning_rate = 0.01

    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_x, train_y = build_dataset(train_sample)
    log = []

    for epoch in range(epoch_num):
        model.train()
        total_loss = []
        for i in range(0, train_sample, batch_size):
            x_batch = train_x[i:i+batch_size]
            y_batch = train_y[i:i+batch_size]
            loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss.append(loss.item())
        print("第{}轮 平均Loss: {:.4f}".format(epoch+1, np.mean(total_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(total_loss))])

    torch.save(model.state_dict(), "multi_model.bin")

    # 可视化
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用模型预测
def predict(model_path, input_vec):
    model = TorchModel(input_size=5, num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(input_vec)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
    for vec, prob, pred in zip(input_vec, outputs, preds):
        print(f"输入: {vec}, 预测类别: {pred.item()}, 概率分布: {prob.numpy()}")

if __name__ == "__main__":
    # main()
    test_vec = [
        [0.1, 0.2, 0.3, 0.9, 0.4],  # 类别 3
        [0.99, 0.2, 0.3, 0.1, 0.0],  # 类别 0
        [0.1, 0.9, 0.8, 0.2, 0.3],   # 类别 1
        [0.4, 0.3, 0.6, 0.2, 0.7]    # 类别 4
    ]
    predict("multi_model.bin", test_vec)
