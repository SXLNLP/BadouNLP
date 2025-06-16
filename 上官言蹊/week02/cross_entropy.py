
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义模型
class CrossEntropyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CrossEntropyModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)  # 计算损失
        else:
            pred = torch.argmax(logits, dim=1)  # 获取预测类别
            return pred

# 构建数据集
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []
    for i in range(total_sample_num):
        x = np.random.randn(input_size)  # 生成五维随机向量
        y = np.argmax(x)  # 最大数字所在维度作为类别
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 评估模型
def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        correct = (pred == test_y).sum().item()
        acc = correct / len(test_y)
        return acc

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = CrossEntropyModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集和测试集
    train_x, train_y = build_dataset(train_sample, input_size)
    test_x, test_y = build_dataset(1000, input_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, test_x, test_y)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "cross_entropy_model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

# 预测函数
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = CrossEntropyModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        input_tensor = torch.FloatTensor(input_vec)
        result = model(input_tensor)  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, res))

if __name__ == '__main__':
    main()
    # 示例预测
    test_input = np.random.randn(3, 5)
    predict("cross_entropy_model.bin", test_input)
