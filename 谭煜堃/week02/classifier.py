# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from numpy import array

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：
x是一个5维向量，
如果x的第1维最大，则属于为类别1
如果x的第2维最大，则属于为类别2
如果x的第3维最大，则属于为类别3
如果x的第4维最大，则属于为类别4
如果x的第5维最大，则属于为类别5
"""



# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，并且按照one-hot编码的方式将其归为5个类别
def build_sample():
    x = np.random.rand(5)
    if x[0] == max(x) :
        return x, array([1, 0, 0, 0, 0])
    elif x[1] == max(x) :
        return x, array([0, 1, 0, 0, 0])
    elif x[2] == max(x) :
        return x, array([0, 0, 1, 0, 0])
    elif x[3] == max(x) :
        return x, array([0, 0, 0, 1, 0])
    else :
        return x, array([0, 0, 0, 0, 1])

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 先将列表转换为numpy数组，再转换为tensor
    X = np.array(X)
    Y = np.array(Y)
    # 将numpy数组转换为tensor
    return torch.FloatTensor(X), torch.FloatTensor(Y)



class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 5)  # 线性层
        #self.linear2 = nn.Linear(5, 5)  # 线性层
        self.activation = torch.softmax  # softmax归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 5)
        #x = torch.relu(x) # 第一层的激活函数

        #x = self.linear2(x)  # 第二层线性层(batch_size, 5) -> (batch_size, 5)
        y_pred = self.activation(x,dim= 1 )  # 第二层激活函数（softmax归一化）(batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果



# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本，%d个1类样本，%d个2类样本，%d个3类样本，%d个4类样本，%d个5类样本" % (test_sample_num, sum(y[:,0]), sum(y[:,1]), sum(y[:,2]), sum(y[:,3]), sum(y[:,4])))
    correct, wrong = 0, 0
    # 使用torch.no_grad()，关闭梯度计算功能（因为模型训练时需要计算梯度，而在测试时不需要计算梯度）
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p.tolist().index(max(y_p)) == y_t.tolist().index(max(y_t)):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 400  # 每次训练样本个数
    train_sample = 100000  # 每轮训练总共训练的样本总数
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
            watch_loss.append(loss.item()) #item方法可以获取张量的具体数值
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
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重（weights_only=True保证了只加载权重，不加载其他可能包含恶意代码的对象）
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        input_vec = np.array(input_vec)
        # 确保输入张量是2维的
        if input_vec.ndim == 1:
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0)
        elif input_vec.ndim == 2:
            input_tensor = torch.FloatTensor(input_vec)
        else:
            raise ValueError("输入向量维度必须为1或2")
        result = model.forward(input_tensor)  # 模型预测
        # 打印样本真实值
        print("样本真实值：")
        print(input_tensor)
        input_list=input_tensor.tolist()
        for i in range(input_tensor.shape[0]):
            print(f"第{i}个样本的真实值：")
            print(input_list[i].index(max(input_list[i])))
        # 打印预测结果
        print("预测结果：")
        print(result)
        result_list=result.tolist()
        for i in range(result.shape[0]):
            print(f"第{i}个样本的预测值：")
            print(result_list[i].index(max(result_list[i])))
    # for vec, res in zip(input_vec, result):  
    #     print(res)
    #     print(vec)
    #     print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


def test():
    predict("model.bin", [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.9, 0.2, 0.4, 0.5]])


if __name__ == "__main__":
    main()
    test()
