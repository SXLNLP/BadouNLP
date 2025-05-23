# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出层维度=类别数
        # 注意：CrossEntropyLoss 内部已包含Softmax，无需额外添加激活函数

    def forward(self, x, y=None):
        logits = self.linear(x)  # 直接输出logits（未归一化的概率）
        if y is not None:
            # 计算交叉熵损失
            loss = nn.functional.cross_entropy(logits, y)
            return loss
        else:
            return logits  # 预测时返回原始输出



def build_sample():
    x = np.random.random(5)  # 生成五维随机向量
    y = np.argmax(x)  # 最大值的索引作为类别标签（0~4）
    return x, y


def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 标签直接保存为整数（无需one-hot）
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意标签是LongTensor



def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        logits = model(x)  # 获取模型输出（未归一化）
        y_pred = torch.argmax(logits, dim=1)  # 取概率最大的索引作为预测类别
        correct = torch.sum(y_pred == y).item()
    acc = correct / test_sample_num
    print(f"正确预测数：{correct}, 准确率：{acc:.4f}")
    return acc



def main():
    # 超参数配置
    input_size = 5
    num_classes = 5  # 类别数=5
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    learning_rate = 0.001

    # 初始化模型和优化器
    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 生成训练数据
    train_x, train_y = build_dataset(train_sample)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):

            start = batch * batch_size
            end = start + batch_size
            x = train_x[start:end]
            y = train_y[start:end]


            loss = model(x, y)

            # 反向传播
            loss.backward()
            optim.step()
            optim.zero_grad()

            watch_loss.append(loss.item())

        # 评估并记录日志
        avg_loss = np.mean(watch_loss)
        print(f"第{epoch + 1}轮平均loss：{avg_loss:.4f}")
        acc = evaluate(model)
        log.append([acc, avg_loss])


    torch.save(model.state_dict(), "model.bin")
    plt.plot([l[0] for l in log], label="Accuracy")
    plt.plot([l[1] for l in log], label="Loss")
    plt.legend()
    plt.show()



def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        probs = torch.softmax(logits, dim=1)  # 手动计算概率分布
        y_pred = torch.argmax(logits, dim=1)

    for vec, prob, pred in zip(input_vec, probs, y_pred):
        print(f"输入：{vec} => 预测类别：{pred.item()}, 各类概率：{prob.numpy().round(3)}")



if __name__ == "__main__":
    main()

