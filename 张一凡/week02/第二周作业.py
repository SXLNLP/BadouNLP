# coding:utf8

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 第一步，定义使用哪些网络层
        self.linear = nn.Linear(input_size, input_size)  # 线性层，输出维度output_size，对应5个类别
        # 无需使用激活函数，torch内部cross_entropy已经包含了Softmax操作
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 第二步，定义这些网络层是如何组合的
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, input_size)
        if y is not None:
            return self.loss(x, y.squeeze().long())  # 计算交叉熵损失
        else:
            return torch.softmax(x, dim=-1)  # 输出概率分布


# 生成一个样本，最大的数字所在的维度决定其类别
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 最大值的索引作为类别
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意Y改为LongTensor

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_counts = [0] * 5
    for label in y:
        class_counts[label.item()] += 1
    print("本次预测集中各类别样本数量:", class_counts)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 获取概率分布
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测类别
        for y_p, y_t in zip(predicted_classes, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


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
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()
        print(f"输入：{vec}, 预测类别：{predicted_class}, 各类别概率：{res.numpy().round(4)}")

if __name__ == "__main__":
    main()
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],  # 类别0概率最高
        [0.1, 0.8, 0.2, 0.3, 0.4],  # 类别1概率最高
        [0.2, 0.5, 0.7, 0.3, 0.4],  # 类别2概率最高
        [0.3, 0.6, 0.3, 0.7, 0.4],  # 类别3概率最高
        [0.4, 0.7, 0.3, 0.8, 0.9]   # 类别4概率最高
    ]
    predict("model.bin", test_vec)
