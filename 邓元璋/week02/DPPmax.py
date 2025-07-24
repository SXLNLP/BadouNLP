import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第i维(0<i<1)大就把向量归为第i类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        """
        初始化神经网络模型
        input_size: 输入特征维度
        output_size: 输出类别维度
        """
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性变换层
        self.activation = torch.sigmoid  # 激活函数使用sigmoid
        self.loss = nn.functional.cross_entropy # 使用交叉信息熵损失函数

    def forward(self, x, y=None):
        """
        前向传播函数
        x: 输入数据
        y: 标签数据(训练时使用)
        """
        x = self.linear(x)  # 线性变换
        y_pred = self.activation(x)  # 应用激活函数
        if y is not None:  # 如果提供了标签，则计算损失
            return self.loss(y_pred, y)
        else:  # 否则返回预测结果
            return y_pred


def build_sample():
    """
    生成单个样本
    输入: 5个随机数组成的向量
    输出: one-hot向量，最大值所在位置为1，其余为0
    """
    x = np.random.random(5)
    y = np.zeros((1,5))
    y[0][x.argmax()]=1
    return x, y[0]

def build_dataset(total_sample_num):
    """
    生成指定数量的样本数据集
    total_sample_num: 样本总数
    返回: 特征张量和标签张量
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def evaluate(model):
    """
    评估模型性能
    model: 训练好的模型
    返回: 准确率
    """
    model.eval()  # 设置为评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():  # 不计算梯度，节省资源
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if y_p.numpy().argmax() == y_t.numpy().argmax():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    """
    模型训练主函数
    """
    # 训练超参数设置
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 批次大小
    train_sample = 50000  # 训练样本数
    input_size = 5  # 输入维度
    output_size = 5  # 输出维度
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, output_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []  # 记录训练过程

    # 生成训练数据
    train_x, train_y = build_dataset(train_sample)

    # 开始训练
    for epoch in range(epoch_num):
        model.train()  # 设置为训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 获取当前批次数据
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            # 前向传播计算损失
            loss = model(x, y)

            # 反向传播更新参数
            loss.backward()
            optim.step()
            optim.zero_grad()

            watch_loss.append(loss.item())

        # 打印本轮训练信息
        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 评估模型
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 绘制训练过程曲线
    plt.plot(range(len(log)), [l[0] for l in log], label="acc", linestyle="--", marker="o", color="#eb2d2e")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss", linestyle="-", marker="*", color="#00a04e")
    plt.ylim(0, 2)
    plt.xticks(np.arange(0, len(log)))
    plt.title("The loss and acc of DPPmax network", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.legend(loc="best")
    plt.xlabel("Training Frequency ", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.ylabel("Loss and acc ", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.savefig("DPPmax_network.png", dpi=1200)  # 保存图片
    plt.show()
    return


def predict(model_path, input_vec):
    """
    使用训练好的模型进行预测
    model_path: 模型路径
    input_vec: 输入向量列表
    """
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    print(result)
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, res.argmax(), res.max()))


if __name__ == "__main__":
    main()  # 训练模型

    # 测试预测功能
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.bin", test_vec)

