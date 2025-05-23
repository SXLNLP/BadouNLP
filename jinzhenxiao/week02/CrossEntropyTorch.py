# coding:utf8

import numpy as np
import torch
import torch.nn as nn

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律（根据build_sample实现）：x是一个5维向量，其最大元素所在的索引（0-3）决定其类别，
若无单个最大元素或最大元素为x[4]（或前四者均非严格最大），则为类别4。


"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            # 使用softmax 平均概率
            return torch.softmax(x, -1)  # 输出预测结果


def build_sample():
    x = np.random.random(5)
    if x[0] > x[1] and x[0] > x[2] and x[0] > x[3] and x[0] > x[4]:
        return x, [1.0, 0.0, 0.0, 0.0, 0.0]
    elif x[1] > x[0] and x[1] > x[2] and x[1] > x[3] and x[1] > x[4]:
        return x, [0.0, 1.0, 0.0, 0.0, 0.0]
    elif x[2] > x[0] and x[2] > x[1] and x[2] > x[3] and x[2] > x[4]:
        return x, [0.0, 0.0, 1.0, 0.0, 0.0]
    elif x[3] > x[0] and x[3] > x[1] and x[3] > x[2] and x[3] > x[4]:
        return x, [0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        return x, [0.0, 0.0, 0.0, 0.0, 1.0]



# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    X_np = np.array(X, dtype=np.float32)
    Y_np = np.array(Y, dtype=np.float32)

    return torch.from_numpy(X_np), torch.from_numpy(Y_np)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    true_classes_indices = torch.argmax(y, dim=1)

    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        predicted_classes_indices = torch.argmax(y_pred, dim=1)
        correct = (predicted_classes_indices == true_classes_indices).sum().item()

    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}/{test_sample_num}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    print("模型已保存到 model.bin")

    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size, 5)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重

    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

    predicted_classes = torch.argmax(result, dim=1)  # 获取预测类别索引

    for i, vec in enumerate(input_vec):
        pred_class = predicted_classes[i].item()
        probs_list = result[i].tolist()
        class_prob = probs_list[pred_class]

        print(f"输入：{vec}, "
              f"预测类别：{pred_class}, "
              f"该类别概率：{class_prob:.4f}, "
              f"所有类别概率：{[f'{p:.2f}' for p in probs_list]}")

if __name__ == "__main__":
    # main()
    print("\n开始使用保存的模型进行预测:")
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 预计 Class 4
                [0.74963533, 0.5524256, 0.05758807, 0.05520434, 0.04890681],  # 预计 Class 0 (因为第一个最大)
                [0.1, 0.2, 0.8, 0.3, 0.1],  # 预计 Class 2
                [0.1, 0.9, 0.2, 0.3, 0.4]]  # 预计 Class 1
    predict("model.bin", test_vec)
