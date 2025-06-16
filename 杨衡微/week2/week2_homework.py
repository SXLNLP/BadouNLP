# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，找出其中最大值的位置，例如：
如果第1个值最大，则输出 [1,0,0,0,0]
如果第2个值最大，则输出 [0,1,0,0,0]
以此类推
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        # 更改为两层网络，以增强模型表达能力
        self.linear1 = nn.Linear(input_size, 16)  # 第一个线性层，输出16维
        self.linear2 = nn.Linear(16, output_size)  # 第二个线性层，输出5维
        self.relu = nn.ReLU()  # 使用ReLU激活函数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y_true=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 16)
        x = self.relu(x)     # 应用ReLU激活函数
                             # (batch_size, 16) -> (batch_size, 16)
        y_predict = self.linear2(x)  # (batch_size, 16) -> (batch_size, output_size)
        
        if y_true is not None:
            # 交叉熵损失需要的是类别索引而不是one-hot编码
            # y应该是形状为(batch_size,)的张量，包含的是类别索引(0-4)
            return self.loss(y_predict, y_true)  # 预测值和真实值计算损失
        else:
            # (batch_size, output_size=5) 里的 每一行 所有数字相加等于1
            return torch.softmax(y_predict, dim=1)  # 输出预测结果的概率分布


# 生成一个样本
# 随机生成一个5维向量，找出最大值的位置
def build_sample():
    """
    生成一个样本
    随机生成一个5维向量，返回向量和最大值的位置
    Returns:
        x: 5维向量 [0.1, 0.35, 0.5, 0.6, 0.8]
        y: 最大值的位置(0-4)
    """
    x = np.random.random(5)
    # 找出最大值的位置
    y = np.argmax(x)
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 直接存储类别索引数字, 如果第一个数最大，存储索引0, 而并非[1,0,0,0,0]
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签需要是LongTensor类型


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()  # 设置模型为评估模式
    test_sample_num = 100  # 测试样本数
    x, y = build_dataset(test_sample_num)  # 生成测试数据
    
    # 统计各类别的样本数
    class_counts = [0] * 5
    for label in y:
        class_counts[label] += 1
    print("本次预测集中各类别样本数:", class_counts)
    
    correct, wrong = 0, 0
    with torch.no_grad():  # 不计算梯度，节约内存
        y_pred = model(x)  # 模型预测
        # 获取最大概率的位置
        _, predicted = torch.max(y_pred, 1)
        for y_p, y_t in zip(predicted, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 预测正确
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
    output_size = 5  # 输出向量维度
    learning_rate = 0.001  # 学习率
    
    # 建立模型
    model = TorchModel(input_size, output_size)
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
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model_max_value.bin")
    
    # 画图
    print(log)
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画acc曲线
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print("模型参数:", model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    
    # 获取每个输入向量的最大概率位置和概率值
    for i, (vec, res) in enumerate(zip(input_vec, result)):
        # 找出最大概率的位置
        max_idx = torch.argmax(res).item()
        max_prob = res[max_idx].item()
        # 找出真实的最大值位置
        true_max_idx = np.argmax(vec)
        
        print(f"样本 {i+1}:")
        print(f"  输入向量: {vec}")
        print(f"  真实最大值位置: {true_max_idx}")
        print(f"  预测最大值位置: {max_idx}")
        print(f"  预测概率分布: {res.numpy()}")
        print(f"  是否预测正确: {'是' if max_idx == true_max_idx else '否'}")
        print("----------------------------")


if __name__ == "__main__":
    main()
    # 测试样例
    test_vec = [
        [0.7, 0.2, 0.1, 0.3, 0.4],  # 第1个值最大
        [0.2, 0.8, 0.1, 0.3, 0.4],  # 第2个值最大
        [0.2, 0.1, 0.9, 0.3, 0.4],  # 第3个值最大
        [0.2, 0.1, 0.3, 0.8, 0.4],  # 第4个值最大
        [0.2, 0.1, 0.3, 0.4, 0.7]   # 第5个值最大
    ]
    predict("model_max_value.bin", test_vec)
