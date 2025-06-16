# coding:utf8

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""

基于pytorch框架编写模型训练
实现一个自行构造的多分类任务：
规律：五维向量中最大值所在的维度即为类别（0~4）

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5个类别的logits
        self.loss = nn.CrossEntropyLoss()     # 交叉熵损失函数

    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5)
        if y is not None:
            return self.loss(logits, y)  # 直接计算损失
        else:
            return torch.softmax(logits, dim=1)  # 返回概率分布

# 生成一个样本，类别为最大值所在的维度
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index

# 生成一批样本
def build_batch_sample(total_sample_num):
    batch_x = []
    batch_y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        batch_x.append(x)
        batch_y.append(y)
    return torch.FloatTensor(batch_x), torch.LongTensor(batch_y)

# 评估模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_batch_sample(test_sample_num)
    with torch.no_grad():
        probs = model(x)
        predicted_classes = torch.argmax(probs, dim=1)
        correct = (predicted_classes == y).sum().item()
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

# 模型训练
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_batch_sample(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss: {avg_loss:.6f}")
        acc = evaluate(model)
        log.append([acc, avg_loss])

    torch.save(model.state_dict(), "model.bin")

    # 绘图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy / Loss")
    plt.title("Multi-class Classification with CrossEntropyLoss")
    plt.show()

# 使用训练好的模型进行预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(input_vec))
        predicted_classes = torch.argmax(probs, dim=1)
    for vec, prob, pred_class in zip(input_vec, probs, predicted_classes):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class.item(), prob[pred_class].item()))

if __name__ == "__main__":
    main()
    test_vec = [
        [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]
    ]
    predict("model.bin", test_vec)
