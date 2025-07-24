import numpy as np
import torch
import torch.nn as nn


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.fc1(x)
        x = self.activation(x)
        y_pred = self.fc2(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 构造数据data、target
def build_sample():
    x = np.random.random(5)
    y = x.argmax()
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        _, predicted = y_pred.max(1)

        for y_p, y_t in zip(predicted, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)), '\n')
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    input_size = 5
    learning_rate = 0.001

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            loss = model(x, y.reshape(-1))  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果

    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print(f"输入：{vec}, 预测类别:{res}" )  # 打印结果


if __name__ == "__main__":
    main()
    # model_path = "model.bin"
    # input_vec = [[1, 2, 3, 4, 5]]
    # predict(model_path, input_vec)