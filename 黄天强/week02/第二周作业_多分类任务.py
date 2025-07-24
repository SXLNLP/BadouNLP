from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class TorchModel(nn.Module):
    def __init__(self, input_size=5, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        y = self.net(x)
        return y


def build_sample():
    """
    生成一个样本。
    样本的生成方法，代表了我们要学习的规律。
    随机生成一个5维向量，五维随机向量最大的数字在哪维就属于哪一类。
    """
    x = torch.rand(5)
    return x, torch.argmax(x)


def build_dataset(total_sample_num):
    """
    随机生成一批样本，均匀生成。
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.stack(X), torch.stack(Y)


def evaluate(model, test_sample_num: int = 100):
    """
    测试每轮模型的准确率。
    """
    model.eval()
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.softmax(y_pred, dim=1)
        index = torch.argmax(y_pred, dim=1)
        correct = (index == y).sum().item()
        accuracy = correct / test_sample_num
        return accuracy


def main():
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.001  # 学习率

    model = TorchModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    log = {}

    for epoch in range(epoch_num):
        train_x, train_y = build_dataset(train_sample)  # 每轮重新生成数据
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            watch_loss.append(loss)

        mean_loss = torch.stack(watch_loss).mean().item()
        acc = evaluate(model)  # 测试本轮模型结果

        print(f"epoch: {epoch + 1}, loss: {mean_loss:.4f}, acc: {acc:.4f}")

        log[f"{epoch}"] = [acc, mean_loss]
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    x = range(len(log))
    plt.plot(x, [v[0] for v in log.values()], label="acc")  # 画acc曲线
    plt.plot(x, [v[1] for v in log.values()], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    """
    使用训练好的模型做预测。
    """
    model = TorchModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(input_vec)  # 模型预测
        result = torch.softmax(result, dim=1)
    for vec, res in zip(input_vec, result):
        label = torch.argmax(vec).item()
        index = torch.argmax(res).item()
        is_correct = label == index
        mark = '✅' * is_correct + '❌' * (not is_correct)
        print("输入：%s, 实际类别：%d, 预测类别：%d, 概率值：%f  %s" % (vec, label, index, res[index], mark))


if __name__ == "__main__":
    main()

    print("=========")
    test_vec = torch.rand(5, 5)
    predict("model.pth", test_vec)
