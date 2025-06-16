# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大数字在哪个维度就属于哪一类

"""
CLASS_COUNT = 5

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 隐藏层 hidden_layer=128
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(128, CLASS_COUNT)  # 输出层（5个分类）
        self.loss = nn.functional.cross_entropy  # loss函数采用cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.fc1(x)  # (batch_size, input_size) → (batch_size, 128)
        x = self.relu(x)  # 应用ReLU激活函数
        y_pred = self.fc2(x)  # (batch_size, 128) → (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 随机生成一个5维向量，选取值最大的做为该分类，y是类别索引
def build_sample():
    x = np.random.random(CLASS_COUNT)
    y = np.argmax(x)  # 直接获取最大值的索引（0~4）
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 直接存储整数，而非列表
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签为LongTensor

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num) # y是类别索引（0~4），形状为(batch_size,)
    class_counts = {i: (y == i).sum().item() for i in range(CLASS_COUNT)}
    print("各类别样本数量:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        probabilities = torch.softmax(y_pred, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)  # 预测类别索引

        for y_pred, y_true in zip(predicted_classes, y):
            if y_pred == y_true:
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    # train_sample = 5  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # # 建立模型
    model = TorchModel(input_size)
    # # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
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
    plt.show(block=False)
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = torch.nn.functional.softmax(model.forward(torch.FloatTensor(input_vec)), dim=1)
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        pred_prob = torch.max(res).item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class, pred_prob))


if __name__ == "__main__":
    main()
    test_vec = [
        [0.1, 0.3, 0.5, 0.2, 0.4],  # 最大索引2
        [0.9, 0.8, 0.7, 0.6, 0.5],  # 最大索引0
        [0.2, 0.4, 0.3, 0.9, 0.1]  # 最大索引3
    ]
    predict("model.bin", test_vec)
