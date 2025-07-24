#改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size,num_class=5):
        super(TorchModel, self).__init__()
        # self.linear = nn.Linear(input_size, 1)  # 线性层
        # self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数
        self.linear=nn.Linear(input_size,num_class)
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        # if y is not None:
        #     return self.loss(y_pred, y)  # 预测值和真实值计算损失
        # else:
        #     return y_pred  # 输出预测结果
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y.long().squeeze())
        else:
            return torch.softmax(x, dim=1)  # 预测时返回概率

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个五维数组，并返回数组中最大值所在的维度
def build_sample():
    arr = np.random.random(5)
    # print(arr,np.argmax(arr))
    # 找到最大值的维度
    return arr, np.argmax(arr)

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
    acc = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{acc:.4f}")
    return acc

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size,num_class=5) # 规定输入的是5维的
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


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel(input_size=5, num_class=5)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(input_vec)
        outputs = model(inputs)
        probs, classes = torch.max(outputs, 1)

        for vec, cls, prob in zip(input_vec, classes, probs):
            print(f"输入：{vec}, 预测类别：{cls.item()}, 概率：{prob.item():.4f}")


if __name__ == "__main__":
    main()
