import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import json

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法

"""


# 定义模型Torchmodel()类
class TorchModel(nn.Module):
    def __init__(self, input_size, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(
            input_size, sentence_length + 1
        )  # +1是因为，a不存在时候，输出的特殊类别
        self.loss = nn.CrossEntropyLoss()
        self.embedding = nn.Embedding(len(vocab), input_size)
        self.rnn = nn.RNN(input_size, input_size, batch_first=True)

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(
            x
        )  # 前向传播，run_out 所有时间步的隐藏状态，hidden 最后一个时间步隐藏状态
        last_hidden = hidden.squeeze(
            0
        )  # 取 rnn 的最后时间步的输出，最后一个序列包含整个序列 等同于run_out[:,-1,:]

        y_pred = self.linear(last_hidden)
        if y is not None:
            y = y.long().view(
                -1
            )  # 是标准的标签预处理操作，确保数据类型和形状符合 CrossEntropyLoss 的要求。
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=1)  # 归一化的类别概率


# 为vocab每个字符char,生成一个索引index
def build_vocab():
    chars = "abcdefghijk"  # 字符集
    vocab = {"pad": 0}
    # enumerate() 遍历同时获取索引index和值char
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab


# 随机生成一个样本，random.sample()从词表中，随机选取k个不重复的元素,如果是元素a,输出索引，否则输出一个无效标签
def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    if "a" in x:
        y = x.index("a") + 1
    else:
        y = sentence_length
    x = [
        vocab.get(word, vocab["unk"]) for word in x
    ]  # 将字符序列转换为数字序列，，为了做embedding
    return x, y


# 建立数据集
def build_dataset(total_sample_num, vocab, sentence_length):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 评估函数evaluate,测试每轮的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    total_sum_num = 200
    x, y = build_dataset(total_sum_num, vocab, sample_length)
    print(f"本次测试集中共有：{len(y)} 个样本")
    correct, wrong = 0, 0

    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    acc = correct / (correct + wrong)
    print(f"正确个数：{correct},正确率：{acc:4f}")
    return acc


# 训练过程main()
def main():
    epoch_num = 20  # 训练轮数
    batch_size = 40  # 每次训练样本数
    train_sample = 1000  # 每轮训练总共样本总数
    char_dim = 30  # 每个字的维度
    senten_lenghth = 10  # 样本文本的长度
    learining_rate = 0.01

    vocab = build_vocab()
    model = build_model(vocab, char_dim, senten_lenghth)
    optim = torch.optim.Adam(model.parameters(), lr=learining_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            train_x, train_y = build_dataset(batch_size, vocab, senten_lenghth)
            optim.zero_grad()
            loss = model(train_x, train_y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print(f"第{epoch+1}轮，平均loss:{avg_loss:.4f}")

        acc = evaluate(model, vocab, senten_lenghth)  # 在测试集评估准确率
        log.append([acc, avg_loss])  # 记录准确率和损失值

    # 保存模型参数（state_dict)只保存参数, 不需要 return，或确保它在最后
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


"""     # 绘制准确率和损失曲线
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="avg_loss")  # 损失曲线
    plt.legend()  # 显示图例
    plt.show()  # 显示图像 """


# 使用训练好的模型，做预测,model_path为保存的模型文件的路径，input_vec是预测的输入数据的列表
def predict(model_path, vocab_path, input_vec):
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度

    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))  # 加载保存的数据模型
    x = []
    for input_string in input_vec:
        x.append([vocab[char] for char in input_string])  # 讲输入序列号

    model.eval()  # 设置为评估模式，确保预测结果准确稳定一致
    with torch.no_grad():  # 禁用梯度，节省内存，加速计算
        # prod = model(torch.FloatTensor(input_vec))  # 输入转为张量，并预测
        # pred_classes = torch.argmax(prod, dim=1)  # 取概率最大的类别，输出预测类别的索引
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_vec):
        print(
            f"输入：{input_string},预测类别：{torch.argmax(result[i])},各类概率：{result[i]}"
        )


# 结果
if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict("model.pth", "vocab.json", test_strings)
