# noinspection PyDuplicatedCode
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# 改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
# [0.1, 0.2, 0.3, 0.4, 0.5]，属于分类4


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        # self.activation = nn.Sigmoid() 交叉熵不需要激活函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x


# 创建样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x = np.random.rand(5)
        X.append(x)
        Y.append(np.argmax(x))
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.long)


def evaluate(model):
    model.eval()
    test_x, test_y = build_dataset(100)
    with torch.no_grad():  # 一般model(x, y)会计算梯度
        y_pred = model(test_x)
        y_pred = torch.argmax(y_pred, dim=1)
        correct = (y_pred == test_y).sum().item()
        print(f"100个样本用此模型，成功预测{correct}个，失败预测{100 - correct}个，准确率：{(correct / 100):.6%}")
    return correct / 100


# 训练
def train_dataloader():
    lr = 0.01  # 学习率
    epochs = 100  # 训练轮数
    batch_size = 20  # 一次训练20个样本
    total_sample_num = 5000  # 5000个样本
    log = []
    # 建立模型
    model = TorchModel(5, 5)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # 真实的训练数据
    train_x, train_y = build_dataset(total_sample_num)
    data_set = TensorDataset(train_x, train_y)
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    # 训练过程
    for epoch in range(epochs):
        # 开启训练模式
        model.train()
        loss_list = []
        for x, y in dataloader:
            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算损失
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新梯度
            loss_list.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(loss_list)}")
        # 每轮结束后如果想要测试模型结果
        acc = evaluate(model)
        log.append([acc, np.mean(loss_list)])
    print(log)
    plt.plot(range(len(log)), [i[0] for i in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [i[1] for i in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), f"week2_TorchDemo_work.pth")


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    input_tensor = torch.tensor(input_vec, dtype=torch.float32)
    with torch.no_grad():
        result = model(input_tensor)
        print(result)
        for vec, res in zip(input_tensor, result):
            prob = torch.softmax(res, dim=0)
            print(f"prob:{prob}")
            pred_class = torch.argmax(prob).item()
            pred_prob = prob[pred_class].item()
            print(f"输入：{vec.numpy()}, 预测类别：{pred_class}, 概率值：{pred_prob:.6f}")


def main():
    train_dataloader()


if __name__ == '__main__':
    main()
    # test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #             [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #             [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #             [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    # predict("week2_TorchDemo_work.pth", test_vec)

