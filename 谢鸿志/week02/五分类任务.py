# 五个输入，哪个数最大属于第几类，损失函数用交叉墒
import numpy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，五个输入，哪个数最大属于第几类，损失函数用交叉墒

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，5分类
        self.loss = nn.CrossEntropyLoss()  # loss函数使用交叉熵

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size,input_size)->(batch_size,5)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            y_logits = torch.softmax(y_pred, dim=1)
            predicted_class = torch.argmax(y_logits, dim=1)  # 取输出的五维logits的最大值的索引
            return predicted_class  # 输出预测结果


def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型/实例化模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 每次取一个batch数据更新参数
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            # 计算loss
            loss = model(x, y)
            # 反向传播，计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            optim.zero_grad()  # 梯度归零，防止梯度累积
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        accuracy = evaluate(model)  # 测试本轮模型结果
        log.append([accuracy, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 每轮训练画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    with torch.no_grad():
        preds = model(x)  # shape: [batch_size], 每个是类别索引

    # 初始化每类的计数器
    total_per_class = [0] * 5
    correct_per_class = [0] * 5

    for pred, true in zip(preds, y):
        label = int(true.item())
        total_per_class[label] += 1
        if int(pred.item()) == label:
            correct_per_class[label] += 1

    # 打印每类样本数和准确率
    print("本次预测集中各类别分布与准确率：")
    for i in range(5):
        total = total_per_class[i]
        correct = correct_per_class[i]
        acc = (correct / total) if total > 0 else 0.0
        print(f" - 类别 {i}：{total} 个样本，预测正确 {correct} 个，准确率：{acc:.2%}")

    # 总体准确率
    total_correct = sum(correct_per_class)
    total_samples = sum(total_per_class)
    overall_acc = total_correct / total_samples
    print(f"总体预测准确率：{overall_acc:.2%}")

    return overall_acc


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d " % (vec, res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [
        [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 4
        [0.74963533, 0.55242560, 0.95758807, 0.95520434, 0.84890681],  # 2
        [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],  # 1
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.13588940],  # 2
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.13588940],  # 2
        [0.24685248, 0.80337156, 0.15120206, 0.20991419, 0.53049848],  # 1
        [0.73011541, 0.10258844, 0.62098259, 0.88850194, 0.37117433],  # 3
        [0.09054711, 0.85486849, 0.85498069, 0.46832103, 0.73114310],  # 2
        [0.30819061, 0.94331677, 0.49172152, 0.21169433, 0.41586388],  # 1
        [0.99999991, 0.99999992, 0.99999993, 0.99999994, 0.99999995],  # 4
    ]
    predict("model.bin", test_vec)
