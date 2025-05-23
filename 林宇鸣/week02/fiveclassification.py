import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的五分类 (机器学习) 任务
分类规则：x是一个5维向量，类别由五个元素的大小关系决定
第一类：第一个元素最大
第二类：第二个元素最大
第三类：第三个元素最大
第四类：第四个元素最大
第五类：第五个元素最大
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)
    return x, label

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)


    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted_labels = torch.argmax(y_pred, dim=1)
        correct = (predicted_labels == y).sum().item()
    accuracy = correct / test_sample_num
    print("正确预测个数： %d, 正确率： %f" % (correct, accuracy))
    return accuracy

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001


    model = TorchModel(input_size)

    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size :(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size :(batch_index + 1) * batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("===============\n 第%d轮平均loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

        torch.save(model.state_dict(),  "model3.bin")

        print(log)
        plt.plot(range(len(log)), [l[0] for l in log], label = "acc")
        plt.plot(range(len(log)), [l[0] for l in log], label="acc")
        plt.legend()
        plt.show()



def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    predicted_labels = torch.argmax(result, dim=1)
    for vec, res in zip(input_vec, predicted_labels):
        print("输入： %s， 预测类别： %d" % (vec, res.item()))



if __name__ == "__main__":
    main()
    test_vec = [[0.5, 0.6, 0.7, 0.1, 0.2],
                [0.1, 0.6, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.7, 0.1, 0.5],
                [0.4, 0.5, 0.5, 0.4, 0.6]]
    predict("model3.bin", test_vec)
