import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
(x是一个五维向量，判断最大值，若最大值在第一维则为第一类)

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层

        # 多分类问题应使用softmax激活函数，但PyTorch的CrossEntropyLoss已经包含了softmax，所以这里不需要
        # self.activation=nn.ReLU()
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵
        # self.loss=nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)

        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出预测的概率分布


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，并返回其类别（最大值所在的索引）
def build_sample():
    x = np.random.random(5)
    category = np.argmax(x)  # 获取最大值所在的索引，这就是我们的类别（0-4）

    return x, category



# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 注意这里直接使用类别索引，不需要包装成列表
    return torch.FloatTensor(X), torch.LongTensor(Y)  # Y需要是LongTensor类型


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回概率分布
        y_pred_class = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        for y_p, y_t in zip(y_pred_class, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 80  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
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
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
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
        result = model(torch.FloatTensor(input_vec))  # 模型预测，返回概率分布
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()  # 获取预测的类别索引
        print("输入：%s, 在第：%d类（预测维度）, 概率分布：%s" % (vec, pred_class + 1, res))  # 打印结果，类别+1是维度（1-5）


if __name__ == "__main__":
    main()
    # 测试预测功能
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
