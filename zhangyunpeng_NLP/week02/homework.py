import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，返回最大元素所在的维度(0-4)
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值(概率分布)
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            # 这里的x是模型的原始输出（logits），即未经过 Softmax 激活的线性层输出
            # 因为这里用的交叉熵CrossEntropyLoss会自动对原始输出进行softmax处理
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出概率分布


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 获取最大值的索引作为标签
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    # 这一步的作用是把模型设置为评估模式
    # 在这个模式下会关闭Dropout（Dropout 会以一定概率（如 p=0.5）随机 “丢弃”（置零）神经元的输出，使模型无法过度依赖某些特定神经元，从而减少过拟合）
    # 而我们在测试时需要模型输出确定的结果，因此需要保留所有神经元的输出
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    #仅需前向传播获取预测结果，无需更新参数，因此应关闭梯度
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        y_pred_class = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        for y_p, y_t in zip(y_pred_class, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10000  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 分类类别数
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        #print("=========\n第%d轮所有loss值为:"% (epoch + 1), watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        print("输入：%s, 预测类别：%d, 置信度：%.2f%%" % (vec, pred_class, res[pred_class] * 100))


if __name__ == "__main__":
    main()
    # 测试用例
    # test_vec = [[0.1, 0.5, 0.3, 0.8, 0.2],
    #             [0.9, 0.2, 0.8, 0.3, 0.1],
    #             [0.2, 0.3, 0.5, 0.1, 0.9]]
    # predict("model.bin", test_vec)
