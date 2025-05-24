import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，标签为其中最大值所在的位置，属于5分类问题
需要用到交叉熵损失函数

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.fc1(x)
        x = self.relu(x)
        y_predict = self.fc2(x)
        if y is not None:   # 训练
            return self.loss(y_predict, y)
        else:   # 预测
            return y_predict

def build_example():
    x = np.random.random(5)
    y = torch.tensor(np.argmax(x))
    return x, y

def build_dataset(test_samples):
    X = []
    Y = []
    for i in range(test_samples):
        x, y = build_example()
        X.append(x)
        Y.append(y)
    X_array = np.array(X)
    Y_array = np.array(Y, dtype=np.int64)
    return torch.FloatTensor(X_array), torch.tensor(Y_array)

def evaluate(model, test_X, test_Y):
    model.eval()
    with torch.no_grad():
        y_pred = model(test_X)
        y_pred_class = torch.argmax(y_pred, dim=1)
        correct = (y_pred_class == test_Y).sum().item()
        total = len(test_Y)
        accuracy = correct / total
    print(f"[评估] 准确率: {accuracy:.4f}，正确数: {correct}/{total}")
    return accuracy

def main():
    epoch_num = 100
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    test_x, test_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, test_x, test_y)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    print("=========\n第%d轮平均loss:%.6f" % (epoch_num, log[-1][1]))
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    main()
