import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大的数字在哪维，就属于哪一类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值（原始分数）
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果（原始分数）


# 生成一个样本，样本的生成方法代表了我们要学习的规律
# 随机生成一个5维向量，最大值所在的维度即为类别标签
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 标签为最大值所在的维度
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


# 测试代码，用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 取最大分数对应的类别
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数：{correct}，正确率：{correct / (correct + wrong):.4f}")
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程
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

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 画图

    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.legend()
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
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        result = torch.softmax(result, dim=1)  # 转换为概率分布

    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        print(f"输入：{vec}")
        print(f"预测类别：{pred_class}")
        print(f"各类别概率：{[round(p, 4) for p in res.tolist()]}")
        print("-" * 40)


if __name__ == "__main__":
    main()

    # 测试预测
    # test_vec = [
    #     [0.1, 0.2, 0.9, 0.3, 0.4],  # 最大值在第2维
    #     [0.5, 0.8, 0.3, 0.2, 0.1],  # 最大值在第1维
    #     [0.9, 0.3, 0.5, 0.2, 0.1],  # 最大值在第0维
    #     [0.1, 0.2, 0.3, 0.4, 0.9],  # 最大值在第4维
    #     [0.1, 0.9, 0.3, 0.5, 0.2]  # 最大值在第1维
    # ]
    # predict("model.bin", test_vec)
