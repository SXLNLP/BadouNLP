# coding:utf8
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
五维判断：x是一个5维向量，向量中哪个标量最大就输出哪一维下标

"""

class CrossEntropyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CrossEntropyModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        self.active01 = torch.softmax # 激活函数 TODO为什么加了激活函数后，loss函数下降变慢
        self.loss = nn.functional.cross_entropy # loss函数采用 交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input, target=None):
        y_pred = self.linear(input)  # (batch_size, input_size) -> (batch_size, output_size)
        y_pred = self.active01(y_pred, dim=-1) # (batch_size, output_size) -> (batch_size, output_size)
        if target is not None:
            return self.loss(y_pred, target)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量 x， 以及最大标量所在下标 y
def build_sample():
    x = np.random.random(5)
    # 获取最大值所在的索引
    max_dimension = np.argmax(x)
    return x, max_dimension


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        # Y.append([y])
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y) # 转为张量

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad(): # 中禁用自动求导（Autograd）的功能，从而避免梯度计算和反向传播(反向传播会影响权重)
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1  # 正确预测样本
            else:
                wrong += 1
    rate = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, rate))
    return rate

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = CrossEntropyModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s " % (vec, torch.argmax(res), res))  # 打印结果


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 5  # 输出向量维度
    learning_rate = 0.001  # 学习率 1e-3
    log = []
    # 建立模型
    model = CrossEntropyModel(input_size, output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
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
        print("=========\n 第 %d 轮平均loss: %f " % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_test.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    # test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.9924256,0.95758807,0.95520434,0.84890681],
    #             [0.40797868,0.67482528,0.99625847,0.34675372,0.19871392],
    #             [0.40797868,0.67482528,0.00625847,0.99675372,0.19871392],
    #             [0.59349776,0.59416669,0.92579291,0.00567412,0.9958894]]
    # predict("model_test.bin", test_vec)
