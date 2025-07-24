import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果

def build_sample():
    #x = random.sample(range(1, 101), 5)
    x = np.random.random(5)
    max_index = np.argmax(x)
    y = [0, 0, 0, 0, 0]
    y[max_index] = 1
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def main():
    # 配置参数
    epoch_num = 1000  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.1  # 学习率
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
        #if acc >= 1.0:
            #break
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    #print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def evaluate(model):
    model.eval() #用于将模型切换至评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad(): #禁止梯度计算
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_x in zip(y_pred, x):  # 与真实标签进行对比
            #print(y_x, y_p)
            if np.argmax(y_x) == np.argmax(y_p):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        #result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        result = model.forward(input_vec)  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s,输出值：%s" % (vec,np.argmax(res)))  # 打印结果


if __name__ == "__main__":
    #print("启动训练")
    #main()
    print("开始预测")
    """test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                 [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                 [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                 [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]"""
    test_vec = []
    for i in range(5):
        x, y = build_sample()
        test_vec.append(x)
    predict("model.bin", torch.FloatTensor(test_vec))
