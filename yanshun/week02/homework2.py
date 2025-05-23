# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大的数字在哪维就属于哪一类。

"""

# 可以设置随机种子seed，保证每次的随机数都是一样的
def setup_seed(seed):
    torch.manual_seed(seed)

# 设置随机数种子
setup_seed(42)


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # 线性层
        self.linear2 = nn.Linear(hidden_size, 5)
        self.sig = nn.Sigmoid()  # nn.Sigmoid() sigmoid归一化函数
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 1)
        x = self.sig(x)  # (batch_size, 1) -> (batch_size, 1)
        x = torch.softmax(x, dim=1)
        x = self.linear2(x)
        y_pred = x
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大的数字在哪维就属于哪一类。
def build_sample():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = np.empty((total_sample_num, 5))
    Y = np.empty((total_sample_num, 5))
    for i in range(total_sample_num):
        X[i], Y[i] = build_sample()
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    # 保存为 PyTorch 文件（单个文件包含两个张量）
    torch.save({'X': X, 'Y': Y}, 'dataset.pt')
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, val_x, val_y):
    model.eval()
    x, y = val_x, val_y
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        val_loss = model(x, y).item()
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print(type(np.argmax(y_p)))  # <class 'torch.Tensor'> 隐式转换
            if y_t[torch.argmax(y_p)] == 1:
                correct += 1  # 分类正确
            else:
                wrong += 1  #分类错误
    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong)}, loss: {val_loss}")
    return correct / (correct + wrong), val_loss


def main():
    # 配置参数
    epoch_num = 1000  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    all_sample = 10000  # 每轮训练总共训练的样本总数
    train_sample = 7000
    input_size = 5  # 输入向量维度
    hidden_size = 5
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    # build_dataset(all_sample)
    loaded_data = torch.load('dataset.pt', weights_only=True)
    X, Y = loaded_data['X'], loaded_data['Y']


    train_x = X[:7000]
    train_y = Y[:7000]
    val_x = X[7000:9000]
    val_y = Y[7000:9000]
    test_x = X[9000:]
    test_y = Y[9000:]

    for i in range(5):
        print(f"训练集中共有{int(train_y.sum(dim=0)[i])}个第{i+1}类样本")

    for i in range(5):
        print(f"预测集中共有{int(val_y.sum(dim=0)[i])}个第{i+1}类样本")

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = np.empty(train_sample // batch_size)
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss[batch_index] = loss.item()
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc, val_loss = evaluate(model, val_x, val_y)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss)), val_loss])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.plot(range(len(log)), [l[2] for l in log], label="val_loss")  # 画val_loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
