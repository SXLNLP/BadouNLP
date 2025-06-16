# coding:utf8

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：五维随机向量最大的数字在哪维就属于哪一类。

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)  # 线性层1
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, input_size)
        self.activation = nn.ReLU()
        self.loss = nn.functional.cross_entropy  # loss函数采用

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        y_pred = self.linear3(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return nn.functional.softmax(y_pred, dim=1)  # 输出预测结果


def build_sample(datasize: int) -> (np.ndarray, np.ndarray):
    """
    生成数据集
    :param datasize: 数据集的大小
    """
    return np.random.uniform(0, 100, size=(datasize, 5))


def build_target(x: (np.ndarray, np.ndarray)) -> np.ndarray:
    """
    生成标签集
    五维随机向量最大的数字在哪维就属于哪一类。
    :param x: 输入矩阵
    :return target: 标签集
    """
    target = []
    for i in range(x.shape[0]):
        target.append(np.argmax(x[i]).item())
    return np.array(target)


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    sample_data = build_sample(total_sample_num)
    target_data = build_target(sample_data)
    return torch.FloatTensor(sample_data).to(device), torch.LongTensor(target_data).to(device)


# 使用训练好的模型做预测
def predict(model_path, x: torch.tensor, y: torch.tensor):
    input_size = 5
    model = TorchModel(input_size).to(device)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    correct, wrong = 0, 0
    with torch.no_grad():
        pred_result = model.forward(x)  # 模型预测
        for y_p, y_t in zip(pred_result, y):  # 与真实标签进行对比
            if y_p.argmax(dim=0).item() == y_t:  # 当y_p的概率最大值对应的索引与y_t（真实标签）相同，则认为预测正确
                correct += 1  # 判断正确
            else:
                wrong += 1
        for i in range(len(x)):
            print(
                f"预测集合中第{i}个样本为：{x[i]},预测类别为{pred_result[i].argmax(dim=0).item() + 1},实际类别为{y[i].item() + 1}")
    print(f"正确预测个数：{correct}", f"正确率：{100 * correct / (correct + wrong) :.2f}%")


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model) -> float:
    model.eval()
    test_sample_num = 2000
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred: torch.Tensor = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p.argmax(dim=0).item() == y_t.item():  # 当y_p的概率最大值对应的索引与y_t（真实标签）相同，则认为预测正确
                correct += 1  # 判断正确
            else:
                wrong += 1
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度，因为生成数据集时是5维向量，这里直接写死
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size).to(device)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    with tqdm(total=epoch_num) as t:
        for epoch in range(epoch_num):
            model.train()
            watch_loss = []
            for batch_index in range(train_sample // batch_size):
                x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
                y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                loss = model(x, y)  # 计算loss  model.forward(x,y)
                loss.backward()  # 计算梯度
                optim.step()
                optim.zero_grad()  # 梯度归零
                watch_loss.append(loss.item())

            acc = evaluate(model)  # 测试本轮模型结果
            mean_loss = float(np.mean(watch_loss))
            t.set_postfix(acc=acc, loss=mean_loss)
            log.append([acc, mean_loss])
            t.update(1)
            # 如果loss反复震荡，就提前结束训练
            if len(log) > 100:
                loss_last = np.array([last_log[1] for last_log in log[-10:]]).mean()
                if np.abs((loss_last - mean_loss) / loss_last) < 0.0075:
                    print("loss在10轮内反复震荡，提前结束训练")
                    break

    # 画图
    plt.title("acc and loss")
    acc_list = np.array([l[0] for l in log])
    loss_list = np.array([l[1] for l in log])

    plt.plot(np.arange(len(acc_list)), acc_list, label="acc")
    plt.plot(np.arange(len(loss_list)), loss_list, label="loss")
    plt.legend()
    plt.show()
    plt.savefig("acc and loss.png")
    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    return


if __name__ == "__main__":
    main()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    predict("model.bin", x, y)
