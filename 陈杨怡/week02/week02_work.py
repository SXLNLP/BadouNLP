import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
# 定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 修改输出维度为5
        self.loss = nn.CrossEntropyLoss  # 使用交叉熵损失函数
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(x, y.squeeze().long())  # 使用 logits 计算损失
        else:
            return x  # 输出 logits
# 生成五维随机向量
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)
    return x, label
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 计算每个类别的样本数量
    category_counts = [sum(y == i for y in y.numpy()) for i in range(5)]
    print("本次预测集中类别0的样本有%d个，类别1的样本有%d个，类别2的样本有%d个，类别3的样本有%d个，类别4的样本有%d个" % tuple(category_counts))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        y_pred_classes = torch.argmax(y_pred, dim=1)  # 将预测值转换为类别索引
        for y_p, y_t in zip(y_pred_classes, y.squeeze()):  # 与真实标签进行对比
            if int(y_p) == int(y_t):
                correct += 1  # 判断正确
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
    model.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
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
if __name__ == "__main__":
    main()
