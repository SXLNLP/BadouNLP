# 构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.LSTM(vector_dim, hidden_size=64, batch_first=True)  # 改用LSTM
        self.classify = nn.Linear(64, sentence_length + 1)  # 输出类别数为长度+1（0-5为位置，6表示不存在）
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch, sen_len) -> (batch, sen_len, dim)
        output, (hn, cn) = self.rnn(x)  # output: (batch, sen_len, hidden)
        # 取每个序列最后一个隐藏状态
        x = output[:, -1, :]  # (batch, hidden)
        y_pred = self.classify(x)  # (batch, 7)
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 需要处理标签形状
        else:
            return torch.softmax(y_pred, dim=1)


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 仅保留英文字符
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 生成包含至少一个a的样本
    has_a = random.random() > 0.3  # 控制a出现概率
    chars = []
    a_pos = sentence_length  # 初始化为超出位置

    for i in range(sentence_length):
        if has_a and a_pos == sentence_length and random.random() < 0.3:  # 确定a的位置
            char = 'a'
            a_pos = i
        else:
            char = random.choice(list(vocab.keys())[1:-1])  # 排除pad和unk
        chars.append(char)

    # 如果未生成a，则随机替换一个位置为a
    if 'a' not in chars and random.random() < 0.5:
        a_pos = random.randint(0, sentence_length - 1)
        chars[a_pos] = 'a'
        has_a = True

    label = a_pos if has_a else sentence_length  # 6表示不存在
    x = [vocab.get(c, vocab['unk']) for c in chars]
    return x, label


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)  # 标签改为Long


def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y).sum().item()
    acc = correct / len(y)
    print(f"正确预测数：{correct}, 正确率：{acc:.4f}")
    return acc


def main():
    epoch_num = 20
    batch_size = 32
    train_sample = 2000
    char_dim = 32
    sentence_length = 6
    learning_rate = 0.001

    vocab = build_vocab()
    model = TorchModel(char_dim, sentence_length, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 生成完整训练集
    train_x, train_y = build_dataset(train_sample, vocab, sentence_length)

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        permutation = torch.randperm(train_sample)  # 打乱顺序

        for i in range(0, train_sample, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = train_x[indices]
            batch_y = train_y[indices]

            optim.zero_grad()
            loss = model(batch_x, batch_y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / (train_sample // batch_size)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        evaluate(model, vocab, sentence_length)

    torch.save(model.state_dict(), "position_model.pth")

    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)  # 确保ASCII关闭以支持中文
    return


def predict(model_path, vocab_path, input_strings):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = TorchModel(32, 6, vocab)
    model.load_state_dict(torch.load(model_path))

    x = []
    for s in input_strings:
        encoded = [vocab.get(c, vocab['unk']) for c in s[:6]]
        x.append(encoded + [0] * (6 - len(encoded)))  # padding

    model.eval()
    with torch.no_grad():
        probs = model(torch.LongTensor(x))
        preds = torch.argmax(probs, dim=1)

    for s, prob, pred in zip(input_strings, probs, preds):
        pos = pred.item()
        print(f"输入：{s.ljust(8)} 预测位置：{pos if pos < 6 else '无'} 置信度：{torch.max(prob):.4f}")


if __name__ == "__main__":
    main()
    test_strings = [
        "a***",
        "bcade",
        "xyz",
        "bbaacc",
        "qweasd",
        "axxxxx"
    ]
    predict("position_model.pth", "vocab.json", test_strings)
