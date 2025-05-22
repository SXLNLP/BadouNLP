# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，y也是一个5维向量，x最大的维度对应的y的维度为1，否则为0

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.GELU(),
            nn.Linear(32,5)
        )
        # 已包含 softmax 激活函数，内部会自动转化 one-hot类型，所以y的真实值不需要是 one-hot类型
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.layer(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim = -1)

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

    X_array = np.array(X)
    Y_array = np.array(Y, dtype=np.int64)


    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_array, dtype=torch.long)

    return X_tensor, Y_tensor


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y_true = build_dataset(test_sample_num)

    # 统计每个类别的样本数
    class_counts = torch.bincount(y_true, minlength=5)
    print("各类别样本数：")
    for i in range(5):
        print(f"类别 {i}: {class_counts[i].item()} 个")

    with torch.no_grad():
        y_prob = model(x)  # 模型预测 model.forward(x)
        y_pred = torch.argmax(y_prob, dim=1)  # 预测类别索引（0~4）
        correct = (y_pred == y_true).sum().item()  # 正确数

    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.0001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model.forward(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
