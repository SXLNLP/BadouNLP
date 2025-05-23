"""
现在有一个多分类任务
输入一个5维向量，输出这个向量属于哪一维（哪一维数字最大就属于哪一维）
1. 准备训练和测试数据
2. 定义模型
3. 进行模型训练
4. 测试数据测试训练出来的模型
5. 生成最后训练以及测试好的模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 生成单条数据,生成的结果是:
# (array([0.96253346, 0.58443919, 0.20423323, 0.69523221, 0.47746297]), 0)
def generate_one_data():
    x = np.random.random(5)
    return x, np.argmax(x)


# 生成一批数据,生成5条结果是：
# (tensor([[0.9694, 0.6270, 0.5036, 0.5051, 0.5460],
#         [0.5842, 0.6094, 0.3437, 0.8455, 0.6890],
#         [0.2153, 0.1516, 0.9776, 0.0014, 0.8346],
#         [0.9268, 0.9299, 0.2972, 0.8904, 0.2191],
#         [0.7882, 0.6108, 0.7007, 0.1040, 0.5504]]), tensor([0., 3., 2., 1., 0.]))
def generate_batch_data(data_size):
    X = []
    Y = []
    for i in range(data_size):
        x, y = generate_one_data()
        X.append(x)
        Y.append(y)
    return (torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y)))


def generate_test_data(data_size):
    X = []
    Y = []
    for i in range(data_size):
        x, y = generate_one_data()
        X.append(x)
        Y.append(y)
    return (torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y)))


# 定义模型，因为是5分类问题，所以用crossEntropy交叉熵损失函数
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)
        # self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    # 看是否输入标签，如果输入标签，那么就进行损失计算；如果不输入，那么直接输出预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        # y_pred = self.activation(x)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y)


# 测试一下模型训练的准确率怎么样
def evaluate_model(model):
    model.eval()
    test_data_num = 100
    test_x, test_y = generate_test_data(test_data_num)
    with torch.no_grad():
        y_pred = model(test_x)
        _, predicted = torch.max(y_pred, 1)
        correct_num = (predicted == test_y).sum().item()
    print(f"正确预测的个数是：{correct_num}, 准确率是：{(correct_num / test_data_num):.2%}")


# 下面开始训练模型
# 一共训练100轮，每轮一共训练5000条数据,每轮每次训练20条数据
'''
1. 建立模型
2. 创建训练数据集
3. 选择优化器
'''


def main():
    epoch_num = 100
    train_data_num = 5000
    batch_size = 20
    learning_rate = 0.1
    input_size = 5

    # 建立模型
    train_model = TorchModel(input_size)
    # 创建训练数据集
    train_x, train_y = generate_batch_data(train_data_num)
    # 优化器
    optim = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    # 开始训练
    for epoch in range(epoch_num):
        train_model.train()
        loss_list = []
        # 按批次取数据，每次都取batch_size个数据
        # 一共分成250批进行训练，每次取20条数据
        for batch_index in range(train_data_num // batch_size):
            # 获取切片
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = train_model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            loss_list.append(loss.item())
        print(f"第 {epoch} 轮平均loss是：{np.mean(loss_list)}")
        if (epoch % 10 == 0):
            evaluate_model(train_model)

    torch.save(train_model.state_dict(), "my_model.bin")


# 用训练好的模型进行预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_tensor = torch.FloatTensor(input_vec)

    with torch.no_grad():
        output = model.forward(input_tensor)  # 原始输出 logits
        probs = F.softmax(output, dim=1)  # 转换为概率
        _, predicted = torch.max(probs, 1)

    for vec, pred, prob in zip(input_vec, predicted, probs):
        prob_percent = prob[pred.item()].item() * 100
        real_class = np.argmax(vec)
        print(f"输入：{vec}, 真实类别：{real_class},预测类别：{pred.item()}, 概率值：{prob_percent:.2f}%")


if __name__ == "__main__":
    # main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("my_model.bin", test_vec)
