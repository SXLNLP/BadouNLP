import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5个单元
        self.loss = nn.CrossEntropyLoss()       # 交叉熵损失

    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, 5)
        if y is not None:
            return self.loss(x, y)
        else:
            return x

def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_counts = [(y == i).sum().item() for i in range(5)]
    print("各类别样本数量：", class_counts)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y).sum().item()
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

def main():
    epoch_num = 40
    batch_size = 40
    train_sample = 10000
    input_size = 5
    learning_rate = 0.001
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), "model.bin")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_classes = torch.argmax(output, dim=1)
        for vec, prob, cls in zip(input_vec, probabilities, predicted_classes):
            print(f"输入：{vec}, 预测类别：{cls.item()}, 各类别概率：{prob.tolist()}")

if __name__ == "__main__":
    main()
    test_vec = [
        [0.9, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.8, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.6, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.5]
    ]
    predict("model.bin", test_vec)
