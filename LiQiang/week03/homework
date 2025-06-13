# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from torch.utils.data import DataLoader, TensorDataset

"""

基于pytorch的RNN网络编写
实现一个多分类任务：预测字符'a'第一次在字符串中出现的位置
如果字符串中没有'a'，则分类为字符串长度+1的位置

"""


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 多分类使用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim)
        x = output[:, -1, :]  # 取最后一个时间步的输出 (batch_size, vector_dim)
        y_pred = self.classify(x)  # (batch_size, num_classes)

        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 输出概率分布


# 字符集
def build_vocab():
    chars = "你我他其他abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机生成字符串
    chars = list(vocab.keys() - {'pad', 'unk'})
    x = [random.choice(chars) for _ in range(sentence_length)]

    # 查找'a'第一次出现的位置
    if 'a' in x:
        y = x.index('a') + 1  # 位置从1开始计数
    else:
        y = sentence_length + 1  # 没有'a'的情况

    x = [vocab.get(char, vocab['unk']) for char in x]  # 转为索引
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y - 1])  # 类别从0开始
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length, num_classes):
    model = RNNModel(char_dim, sentence_length, vocab, num_classes)
    return model


# 测试
def evaluate(model, vocab, sentence_length, num_classes):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)

    # 统计每个类别的样本数
    class_count = [0] * num_classes
    for label in y:
        class_count[label.item()] += 1
    print("各类别样本数:", class_count)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y.squeeze()).sum().item()
        wrong = len(y) - correct

    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 批量大小
    train_sample = 5000  # 训练样本总数
    char_dim = 64  # 字符嵌入维度
    sentence_length = 10  # 字符串长度
    num_classes = sentence_length + 1  # 类别数 (位置1-10 + 无'a'的情况)
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    print("词汇表大小:", len(vocab))

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 准备训练数据
    train_x, train_y = build_dataset(train_sample, vocab, sentence_length)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = model(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"Epoch {epoch + 1}/{epoch_num}, 平均loss: {avg_loss:.4f}")

        # 评估
        acc = evaluate(model, vocab, sentence_length, num_classes)
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return log


# 预测函数
def predict(model_path, vocab_path, input_strings, sentence_length=10):
    char_dim = 64
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    num_classes = sentence_length + 1

    model = build_model(vocab, char_dim, sentence_length, num_classes)
    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 处理输入
    processed = []
    for s in input_strings:
        # 截断或填充到固定长度
        s = s[:sentence_length].ljust(sentence_length, 'x')[:sentence_length]
        encoded = [vocab.get(c, vocab['unk']) for c in s]
        processed.append(encoded)

    x = torch.LongTensor(processed)

    with torch.no_grad():
        probs = model(x)
        predictions = torch.argmax(probs, dim=1) + 1  # 转回1-based索引

    for s, pos in zip(input_strings, predictions):
        if pos == sentence_length + 1:
            print(f"输入: '{s}', 预测: 字符串中没有'a'")
        else:
            print(f"输入: '{s}', 预测: 'a'第一次出现在第{pos}个位置")


if __name__ == "__main__":
    main()
    test_strings = ["abcdefghij", "bcdefghija", "bbb我bbbbb", "aabbccddee", "你yzxyza",'"xyzxyzxyza"', "bbbbbbbbbb"]
    predict("rnn_model.pth", "vocab.json", test_strings)
