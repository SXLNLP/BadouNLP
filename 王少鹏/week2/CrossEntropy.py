# coding=utf-8
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class CrossEntropyModelNumpy:
#     def __init__(self, input_size, num_classes, learning_rate=0.001):
#         self.input_size = input_size
#         self.num_classes = num_classes
#         self.learning_rate = learning_rate
#         self.weights = np.random.randn(input_size, num_classes) * 0.01
#         self.biases = np.zeros(num_classes)
#
#     def forward(self, x):
#         return np.dot(x, self.weights) + self.biases
#
#     def compute_loss(self, logits, y):
#         m = y.shape[0]  # number of samples
#         exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
#         probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # softmax
#         log_likelihood = -np.log(probs[range(m), y])
#         loss = np.sum(log_likelihood) / m
#         return loss, probs
#
#     def backward(self, x, y, probs):
#         m = x.shape[0]
#         y_one_hot = np.zeros(probs.shape)
#         y_one_hot[range(m), y] = 1
#         dlogits = (probs - y_one_hot) / m
#         dweights = np.dot(x.T, dlogits)
#         dbiases = np.sum(dlogits, axis=0)
#         return dweights, dbiases
#
#     def update_parameters(self, dweights, dbiases):
#         self.weights -= self.learning_rate * dweights
#         self.biases -= self.learning_rate * dbiases
#
#
# def build_dataset(total_sample_num, input_size):
#     X = np.random.randn(total_sample_num, input_size)
#     Y = np.argmax(X, axis=1)
#     return X, Y
#
#
# def evaluate(model, test_x, test_y):
#     logits = model.forward(test_x)
#     preds = np.argmax(logits, axis=1)
#     accuracy = np.mean(preds == test_y)
#     return accuracy
#
#
# def main():
#     # Configuration
#     epoch_num = 20
#     batch_size = 20
#     train_sample = 5000
#     input_size = 5
#     num_classes = 5
#     learning_rate = 0.001
#
#     # Initializing the model
#     model = CrossEntropyModelNumpy(input_size, num_classes, learning_rate)
#
#     # Creating training and test sets
#     train_x, train_y = build_dataset(train_sample, input_size)
#     test_x, test_y = build_dataset(50000, input_size)
#
#     log = []
#     for epoch in range(epoch_num):
#         permutation = np.random.permutation(train_sample)
#         train_x_shuffled = train_x[permutation]
#         train_y_shuffled = train_y[permutation]
#
#         for i in range(0, train_sample, batch_size):
#             x_batch = train_x_shuffled[i:i + batch_size]
#             y_batch = train_y_shuffled[i:i + batch_size]
#
#             logits = model.forward(x_batch)
#             loss, probs = model.compute_loss(logits, y_batch)
#             dweights, dbiases = model.backward(x_batch, y_batch, probs)
#             model.update_parameters(dweights, dbiases)
#
#         train_loss, _ = model.compute_loss(model.forward(train_x), train_y)
#         acc = evaluate(model, test_x, test_y)
#         log.append([acc, train_loss])
#         print("Epoch %d: Loss = %.4f, Accuracy = %.4f" % (epoch + 1, train_loss, acc))
#
#     plt.plot(range(len(log)), [l[0] for l in log], label="acc")
#     plt.plot(range(len(log)), [l[1] for l in log], label="loss")
#     plt.legend()
#     plt.show()
#
#
# def predict(model, input_vec):
#     logits = model.forward(input_vec)
#     preds = np.argmax(logits, axis=1)
#     for vec, pred in zip(input_vec, preds):
#         print("输入：{}, 预测类别：{}".format(vec, pred))
#
#
#
#
#
#
# if __name__ == '__main__':
#     main()
#     test_input = np.random.randn(3, 5)
#     model = CrossEntropyModelNumpy(5, 5)
#     predict(model, test_input)
#
#
#

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义模型
class CrossEntropyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CrossEntropyModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

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
    epoch_num = 120
    batch_size = 50
    train_sample = 10000
    input_size = 5
    num_classes = 5
    learning_rate = 0.001
    # 建立模型
    model = CrossEntropyModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集和测试集
    train_x, train_y = build_dataset(train_sample, input_size)
    test_x, test_y = build_dataset(5000, input_size)
    # 训练过程
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
        acc = evaluate(model, test_x, test_y)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 预测函数
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = CrossEntropyModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec)
        result = model(input_tensor)
    for vec, res in zip(input_vec, result):
        print("%s 最大为： %d" % (vec, vec[result]))

if __name__ == '__main__':
    main()
    test_input = np.random.randn(1, 5)
    predict("model.bin", test_input)