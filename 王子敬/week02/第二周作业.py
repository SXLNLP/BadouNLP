import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.liner = nn.Linear(input_size, 5)
        # self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()会在内部自动使用log_softmax所以不需要上面的sigmoid

    # 当输入真实标签时，返回loss值；输入为None时，返回预测值
    def forward(self, x, y=None):
        # 这里取消掉了activation
        y_pred = self.liner(x)
        if y is not None:
            return self.loss(y_pred, y)  # 通过交叉熵使用预测值和真实值计算损失值
        else:
            return y_pred


# 随机生成样本的方法，内容为5维向量，值最大的为正确的类别
def build_random_sample():
    x = np.random.random(5)
    # print(x)
    return x, x.argmax()


# 随机生成样本集
def build_random_dataset(num_of_sample):
    X = []
    Y = []
    for i in range(num_of_sample):
        x, y = build_random_sample()
        X.append(x)
        Y.append(y)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 模型测试，用于测试本周期内模型的准确率
def evaluate(model):
    model.eval()  # 测试
    test_sample_num = 100
    x, y = build_random_dataset(test_sample_num)
    print("正在进行测试")
    # 想通过Counter查询对应类别在数据集中出现的次数，但y这里是floatTensor不能进行操作
    # for i in range(5):
    #     Counter(y)
    #     print("第%d类")
    y_np = y.squeeze().numpy()
    class_count = Counter(y_np)
    # print("测试集中各个类别的数量分布为：")
    # for cls in sorted(class_count.keys()):
    #     print("类别%d有%d个样本。" % (cls+1, class_count[cls]))
    with torch.no_grad():
        y_pred = model(x)  # 使用模型进行预测 等同于model.forward(x)
        prediction = torch.argmax(y_pred, dim=1)
        correct = (prediction == y).sum().item()
        accuracy = correct / test_sample_num
        print("测试的正确率为：%.2f%%" % (accuracy*100))

        return accuracy


def main():
    # 设置模型参数
    epoch_num = 100
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 读入训练集数据
    train_x, train_y = build_random_dataset(train_sample)
    # 查看生成的数据
    # print(train_x)
    # print("-----------------------------------------")
    # print(train_y)
    # 训练模型
    for epoch in range(epoch_num):
        model.train()  # 进入训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):  # 每一个周期训练model的次数为总数据量除以batch_size取整
            # 每次读入的样本为按照当前训练批次按照batch_size进行切片读取操作得到的数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss 操作等同于model.forward(x,y)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())  # 将本次训练的loss存下
        print("----------\n第%d轮平均loss为：%f" % (epoch + 1, np.mean(watch_loss)))  # 对本训练周期的loss求平均
        # 测试本轮模型
        accuracy = evaluate(model)
        log.append([accuracy, float(np.mean(watch_loss))])  # 储存数据用于画图
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)  # 输出测试结果
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()

