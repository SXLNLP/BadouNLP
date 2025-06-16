import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于pytorch的多分类模型训练
规律：x是一个5维向量，最大值的维度索引即为类别（0-4）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5个类别
        self.loss = nn.CrossEntropyLoss()  # loss函数使用交叉熵损失

    # 前向传播
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 直接输出logits (batch_size, 5)
        if y is not None:
            # 将y转换为长整型，并压缩维度以适配CrossEntropyLoss
            return self.loss(y_pred, y.long().squeeze())
        else:
            return y_pred  # 输出原始logits


# 生成一个样本
# 随机生成一个5维向量，找到5维向量最大值的索引作为类别
def build_sample():
    x = np.random.random(5)  # 随机生成一个5维向量
    y = np.argmax(x)  # 取最大值索引
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    y = y.squeeze()  # 压缩维度适配预测结果

    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        prediction = torch.argmax(y_pred, dim=1)  # 取每行最大值的索引
        correct += (prediction == y).sum().item()

    print(" 正确预测个数：%d, 正确率：%f" % ( correct, correct / test_sample_num))
    return correct / (correct + test_sample_num)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
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
            # 获取批次数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
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


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式

    with torch.no_grad():  # 不计算梯度
        y_pred = model(torch.FloatTensor(input_vec))
        result = torch.argmax(y_pred, dim=1)  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 各类别分数：%f" % (vec, res.item(), y_pred.numpy()[0]))  # 打印结果


if __name__ == "__main__":
    main()
