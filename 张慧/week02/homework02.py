# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community.label_propagation import label_propagation_communities

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
"""

class TorchModel(nn.Module):
    def __init__(self, input_size,hidden_size=64,num_classes=4):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # 线性层
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.activation3 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.activation4 = nn.ReLU()
        self.linear5 = nn.Linear(hidden_size, num_classes)
        self.activation5 = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):

        x = self.linear1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.activation2(x)

        x = self.linear3(x)
        x = self.activation3(x)

        x = self.linear4(x)
        x = self.activation4(x)

        x = self.linear5(x)
        y_pred = self.activation5(x)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def build_sample(dim=10):
    """
    生成一个 dim 维向量，并基于新的规则生成四分类标签
    """
    x = np.random.random(dim)  # 随机生成 [0, 1] 区间的 10 维向量

    # 规则 1：sum(x[:3]) > 1.5 且 x[4] - x[5] > 0.2
    if np.sum(x[:3]) > 1.5 and (x[4] - x[5]) > 0.2:
        y = 3

    # 规则 2：x[6] + x[7] > 1.0 且 x[8] * x[9] < 0.5
    elif (x[6] + x[7]) > 1.0 and (x[8] * x[9]) < 0.5:
        y = 2

    # 规则 3：x[1] > 0.7 且 x[2] < 0.3 且 sum(x[3:6]) > 1.0
    elif (x[1] > 0.7) and (x[2] < 0.3) and (np.sum(x[3:6]) > 1.0):
        y = 1

    # 默认类别
    else:
        y = 0

    return x, y

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []

    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])

    return torch.FloatTensor(X), torch.FloatTensor(Y).long().view(-1)  # 将y的维度变为一维

#
# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            predicted_class = torch.argmax(y_p)  # 选择预测的类（概率最大）
            if predicted_class == y_t:  # 判断预测的类别是否与真实标签相同
                correct += 1  # 类别预测正确
            else:
                wrong += 1  # 类别预测错误

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "multi_classification.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = len(input_vec[0])
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        prob, label = torch.max(result, 1)
    for vec, prob,lab in zip(input_vec, prob, label):
        # label = torch.argmax(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, lab.item(), prob.item()))  # 打印结果


if __name__ == "__main__":
    # main()

    # label = [0,2,2,0,1]
    test_vec=[[0.5021045358335918, 0.41627565112692433, 0.10088491818537515, 0.7651044913464449, 0.004022104585643671, 0.6451190944996303, 0.7988995767220287, 0.12613021622578668, 0.24449839049205424, 0.9662155756172425],
              [0.09036816848666207, 0.1054104997695261, 0.27712378457577236, 0.5561619573349053, 0.17425057761282225, 0.5930603750999464, 0.2310130290024608, 0.8093310363423356, 0.1540739580085536, 0.7220690503625439],
              [0.047560741282825614, 0.607570363342176, 0.7100232247787438, 0.986301886918337, 0.6694022895698878, 0.12499429267788753, 0.7103914391227802, 0.5326348535256898, 0.8674510764436348, 0.5025566074369863],
              [0.333319053610879, 0.4749630014742523, 0.03189995760255726, 0.6344632274823281, 0.6664084493278449, 0.41263371528990933, 0.06243228625911312, 0.431729478639843, 0.6367489578974317, 0.872213054677285],
              [0.6709354302603774, 0.8874922447909616, 0.04576596572418323, 0.39408559046391456, 0.7422346629341484, 0.8537868791482262, 0.18935465921929295, 0.14270202037257806, 0.7850937409890606, 0.7995310413714365]]
    
    predict("multi_classification.pth", test_vec)

