"""
第二周作业 from 邵一由

作业描述：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""

import torch         
import torch.nn as nn   
import numpy as np
import random
import json
import matplotlib.pyplot as plt    

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size,output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size,output_size) # 线性层
        # 没写激活函数是因为交叉熵损失函数会自动对x做softmax
        self.loss = nn.functional.cross_entropy #loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self ,x, y = None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x,y) # 预测值和真实值计算损失
        else:
            return torch.softmax(x,dim=1) #输出预测结果

#生成给定数个五维样本，并通np.argmax来返回最大值的索引，返回的是一个标量
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x,y

#随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y) #一开始用FloatTensor报错了，GPT告诉我在用交叉熵的时候索引值数据要求是整形

#测试代码
#用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    class_counts = [sum(y==i).item() for i in range (5)] #计算每个类型的个数，用item（）是为了从torch.Tensor中取得单个数值，不然会得到一个零维张量 很不方便
    print("本次预测集中各类别样本数:", class_counts)
    correct = 0
    with torch.no_grad():
        y_pred = model(x) #模型预测 model.forward(x)
        predicted_classes = torch.argmax(y_pred, dim=1) #预测的数组类别
        correct = (predicted_classes == y).sum().item() #对比预测值和真实值是否一致
    accuracy = correct / test_sample_num
    print(f"正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    #配置参数
    epoch_num = 20 #训练轮数
    batch_size = 20 #每次训练样本个数
    train_sample = 5000 #每轮训练总共训练的样本总数
    input_size = 5 # 输入向量维度
    output_size = 5 #输出向量维度
    learning_rate = 0.001 #学习率
    #建立模型
    model = TorchModel(input_size,output_size)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model(x,y) #计算 loss model.forward(x,y)
            loss.backward() #计算梯度
            optim.step() #更新权重
            optim.zero_grad() #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    #保存模型
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
    output_size = 5
    model = TorchModel(input_size,output_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()  # 获取预测类别索引
        probability = res[predicted_class].item()  # 该类别的预测概率
        print("输入：%s, 预测类别：%d， 概率值：%f" % (vec,predicted_class,probability))

if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
