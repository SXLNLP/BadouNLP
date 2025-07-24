import torch
import torch.nn as nn
import numpy as np


class Week02Module(nn.Module):
    def __init__(self, input_size):
        super(Week02Module, self).__init__()
        self.linear_layer = nn.Linear(input_size, 5)
        self.activation_layer = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

    def forward(self, x, y_true=None):
        linear_output = self.linear_layer(x)
        if y_true is None:
            return self.softmax(linear_output)
        loss = self.activation_layer(linear_output, y_true)
        return loss


def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


def build_dataset(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def test(model, test_sample_size):
    model.eval()
    with torch.no_grad():
        correct_count = 0
        test_dataset_X, test_dataset_Y = build_dataset(test_sample_size)
        for x, y in zip(test_dataset_X, test_dataset_Y):
            pred = model(x)
            print(f"测试数据: {x.tolist()}, 预测值: {pred.tolist()}")
            max_index = np.argmax(pred)
            if max_index == y:
                correct_count += 1
    print("正确预测个数：%d, 正确率：%f" % (correct_count, correct_count / test_sample_size))


def main():
    train_sample_size = 5000
    test_sample_size = 1000
    learning_rate = 0.001
    epoch_num = 1000
    batch_size = 100

    model = Week02Module(5)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_X, train_Y = build_dataset(train_sample_size)

    for epoch in range(epoch_num):
        model.train()
        for batch_index in range(train_sample_size // batch_size):
            x = train_X[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_Y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

    test(model, test_sample_size)


main()
