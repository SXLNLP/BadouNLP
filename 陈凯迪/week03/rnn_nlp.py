#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本a出现的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=64):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)
        # 输出6个类别（位置0-5）
        self.classify = nn.Linear(hidden_size, 6)
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, 6, vector_dim)
        # RNN处理序列
        output, _ = self.rnn(x)  # output: (batch_size, 6, hidden_size)
        # 使用最后一个时间步的输出
        last_output = output[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(last_output)  # (batch_size, 6)

        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 计算损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 输出概率分布


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 确保字符串中一定包含'a'
    # 随机选择'a'出现的位置
    position = random.randint(0, sentence_length - 1)

    # 构建字符串
    x = []
    for i in range(sentence_length):
        if i == position:
            # 在指定位置放置'a'
            x.append('a')
        else:
            # 其他位置随机选择字符（不包括'a'）
            char = random.choice("bcdefghijklmnopqrstuvwxyz")
            x.append(char)

    # 转换为索引
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, position


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    y = y.squeeze()

    # 统计各类别数量
    class_count = [0] * 6
    for label in y:
        class_count[label] += 1
    print("位置分布:", class_count)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 获取概率分布
        predictions = torch.argmax(y_pred, dim=1)  # 获取预测位置

        for pred, true in zip(predictions, y):
            if pred == true:
                correct += 1
            else:
                wrong += 1

    acc = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample = 5000  # 每轮训练样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率

    # 建立字表
    vocab = build_vocab()
    print("字符表:", vocab)

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    # 处理输入
    x = []
    for s in input_strings:
        # 补齐或截断到6个字符
        s = s[:sentence_length].ljust(sentence_length, 'x')
        # 检查是否包含'a'，如果不包含则随机插入一个
        if 'a' not in s:
            pos = random.randint(0, 5)
            s = s[:pos] + 'a' + s[pos + 1:]
        x.append([vocab.get(char, vocab['unk']) for char in s])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        predictions = torch.argmax(result, dim=1)
        probabilities = torch.softmax(result, dim=1)

    position_names = ['首位', '第2位', '第3位', '第4位', '第5位', '第6位']
    for i, s in enumerate(input_strings):
        prob_value = probabilities[i][predictions[i]].item()
        print(f"输入：{s.ljust(6)} => 'a'首次出现在: {position_names[predictions[i].item()]} (概率: {prob_value:.4f})")


if __name__ == "__main__":
    main()
    test_strings = [
        "bcdefa", "aabcde", "xayzza",
        "bbbbbb", "cdefga", "aaaaaa",
        "zzaa", "a", "noahere", "testa"
    ]
    predict("rnn_model.pth", "vocab.json", test_strings)
