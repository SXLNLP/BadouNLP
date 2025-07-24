import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须放在所有库导入之前
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = F.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 创建规律
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y


# 创建数据集
def build_dataset(total_sample_sum):
    X = []
    Y = []
    for i in range(total_sample_sum):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X_np_array = np.array(X)
    Y_np_array = np.array(Y)
    return torch.tensor(X_np_array, dtype=torch.float32), torch.tensor(Y_np_array, dtype=torch.long)


# 测试代码
def test(model):
    # 测试模式
    model.eval()
    test_sample_sum = 100
    x_test, y_test = build_dataset(test_sample_sum)
    # 查看样本分布情况
    unique, counts = np.unique(y_test, return_counts=True)
    converted_dict = dict(zip(unique, counts))
    converted_dict = {int(k): int(v) for k, v in converted_dict.items()}
    print(f'本次预测集中个样本分布-->{converted_dict}')
    correct, worry = 0, 0
    with torch.no_grad():  # 不计算梯度
        y_pred = model(x_test)
        y_pred = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(y_pred, y_test):
            if y_p == y_t:
                correct += 1
            else:
                worry += 1
    print('正确预测个数：%d, 正确率：%f' % (correct, correct / (correct + worry)))
    return correct / (correct + worry)


# 用训练好的模型做训练


def main():
    # 训练次数
    epoch_num = 20
    # 每次训练样本
    batch_num = 20
    # 总训练样本数
    train_num = 5000
    # 输入向量维度
    input_size = 5
    # 学习率
    learning_rate = 0.001
    # 模型训练
    model = TorchModel(input_size)
    # 选择优化器
    optimize = optim.Adam(model.parameters(), lr=learning_rate)
    x_train, y_train = build_dataset(train_num)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_num // batch_num):
            x = x_train[batch_index * batch_num:(batch_index + 1) * batch_num]
            y = y_train[batch_index * batch_num:(batch_index + 1) * batch_num]
            # 计算损失值
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optimize.step()
            # 清零权重
            optimize.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch, np.mean(watch_loss)))
        acc = test(model)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), 'model.bin')
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s" % (vec, np.argmax(res)))  # 打印结果


if __name__ == '__main__':
    main()
    # test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #             [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #             [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #             [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    # predict("model.bin", test_vec)
