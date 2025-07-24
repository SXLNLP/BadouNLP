import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size,output_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return torch.softmax(x, dim=1)

def build_sample():
    x = np.random.rand(5)#生成5个每个元素的值在[0,1)之间
    y = np.argmax(x)  #返回数组x中最大值所在的索引
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X_array = np.array(X, dtype=np.float32)
    Y_array = np.array(Y, dtype=np.int64)   #PyTorch要求CrossEntropyLoss等损失函数强制要求标签为64位整型
    return torch.from_numpy(X_array), torch.from_numpy(Y_array)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计各类别样本数量
    class_counts = [0] * 5
    for label in y:
        class_counts[label] += 1
    print("测试集类别分布:", class_counts)
    correct = 0
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y).sum().item()

    accuracy = correct / test_sample_num
    print(f"正确预测个数: {correct}, 正确率: {accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 5  # 5个类别
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    log = []
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = []
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入，train_x[0:20] train_y[0:20]train_x[20:40]train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            epoch_loss.append(loss.item())
            avg_loss = np.mean(epoch_loss)
        acc = evaluate(model)
        log.append([acc, float(avg_loss)])
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
    torch.save(model.state_dict(), "model.bin")
    print(log)  # 打印完整日志
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 提取所有准确率
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 提取所有loss值
    plt.title('Changes in training process metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))


if __name__ == "__main__":
    main()
