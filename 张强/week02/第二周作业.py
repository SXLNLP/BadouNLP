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
规律：x是一个5维向量，五维随机向量最大的数字在哪维就属于哪一类。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = nn.Softmax(dim=1)  # nn.softmax() softmax激活函数
        self.loss = nn.functional.cross_entropy # loss函数采用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(x, y),y_pred  # 预测值和真实值计算损失
        else:
            return y_pred   # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，向量最大的数字在哪维就属于哪一类。
def build_sample():
    x = np.random.random(5)
    y = np.zeros(5)
    for i in range(5):
        if x[i] == np.max(x):
            y[i] = 1
    return x, y



# 随机生成一批样本
# 正负样本均匀生成
# def build_dataset(total_sample_num):
#     X = []
#     Y = []
#     for i in range(total_sample_num):
#         x, y = build_sample()
#         X.append(x)
#         Y.append([y])
#     return torch.FloatTensor(X), torch.FloatTensor(Y)
def build_dataset(total_sample_num):
    # 预分配 NumPy 数组（假设已知样本形状）
    # 假设单个样本 x 形状为 (feature_dim,)，y 形状为 (1,)
    feature_dim = 5  # 根据实际情况调整
    X_np = np.empty((total_sample_num, feature_dim), dtype=np.float32)
    Y_np = np.empty((total_sample_num, feature_dim), dtype=np.float32)
    for i in range(total_sample_num):
        x, y = build_sample()  # 假设 x 是 np.ndarray，y 是标量或数组
        X_np[i] = x  # 直接填充预分配的数组
        Y_np[i] = y
    print(X_np)
    print(Y_np)
    a = np.sum(Y_np, axis=0)
    print("每一列中1的个数：", a)
    # 一次性转换为张量（零拷贝共享内存）
    X_tensor = torch.from_numpy(X_np)
    Y_tensor = torch.from_numpy(Y_np)
    return X_tensor, Y_tensor,a
# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y , a = build_dataset(test_sample_num)
    print("本次预测集中共有%d个一类样本，%d个二类样本，%d个三类样本，%d个四类样本，%d个五类样本" % (a[0],a[1],a[2],a[3],a[4]))
    # correct, wrong = 0, 1
    with torch.no_grad():
        pred = model(x)  # 模型预测 model.forward(x)1
        breds = torch.argmax(y, dim=1)
        preds = torch.argmax(pred, dim=1)  # 形状 (5,)
        print(breds)
        print(preds)
        # Step 3: 对比预测与真实标签
        correct = (preds == breds)  # 布尔张量，形状 (5,)
        num_correct = correct.sum().item()  # 正确预测数
        print(num_correct)
        total = y.size(0)  # 总样本数

        # Step 4: 计算准确率
        accuracy = num_correct / total
        print(accuracy)
    #     for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
    #         if float(y_p) < 0.5 and int(y_t) == 0:
    #             correct += 1  # 负样本判断正确
    #         elif float(y_p) >= 0.5 and int(y_t) == 1:
    #             correct += 1  # 正样本判断正确
    #         else:
    #             wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (num_correct, accuracy))
    return accuracy


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
    train_x, train_y ,a  = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss,pred = model(x, y)  # 计算loss  model.forward(x,y)
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
        print(result)
    for vec, res in zip(input_vec, result):
        n = torch.argmax(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, n, res[n]))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
