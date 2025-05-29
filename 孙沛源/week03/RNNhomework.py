# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
RNN多分类：预测字母a第一次出现的位置
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
"""

class TorchRNNModel(nn.Module):
    def __init__(self, vocab_size, vector_dim, hidden_dim, sentence_length):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, sentence_length)  # 多分类输出为句子长度种类别
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)                       # (batch, seq_len) -> (batch, seq_len, vector_dim)
        out, h_n = self.rnn(x)                      # out: (batch, seq_len, hidden_dim)
        last_hidden = h_n[-1]                       # 取最后一个时间步隐藏状态 (batch, hidden_dim)
        logits = self.classifier(last_hidden)       # (batch, hidden_dim) -> (batch, sentence_length)
        if y is not None:
            return self.loss_fn(logits, y)
        else:
            return torch.argmax(torch.softmax(logits, dim=1), dim=1)  # 返回预测位置索引


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab["unk"] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    chars = list(vocab.keys())
    chars.remove("pad")
    chars.remove("unk")

    # 确保包含至少一个 'a'
    pos = random.randint(0, sentence_length - 1)
    x = [random.choice(chars) for _ in range(sentence_length)]
    x[pos] = 'a'
    y = pos  # 目标是预测第一次出现 'a' 的位置
    x_ids = [vocab.get(char, vocab["unk"]) for char in x]
    return x_ids, y


def build_dataset(sample_num, vocab, sentence_length):
    dataset_x, dataset_y = [], []
    for _ in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, hidden_dim, sentence_length):
    return TorchRNNModel(len(vocab), char_dim, hidden_dim, sentence_length)


def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    with torch.no_grad():
        y_pred = model(x)
        acc = (y_pred == y).sum().item() / len(y)
    print("预测准确率：%.2f%%" % (acc * 100))
    return acc


def main():
    epoch_num = 50
    batch_size = 32
    train_sample = 1000
    char_dim = 16
    hidden_dim = 64
    sentence_length = 10
    learning_rate = 0.005

    vocab = build_vocab()
    model = build_model(vocab, char_dim, hidden_dim, sentence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for _ in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("第%d轮，平均loss: %.4f" % (epoch + 1, total_loss / (train_sample // batch_size)))
        evaluate(model, vocab, sentence_length)

    torch.save(model.state_dict(), "rnn_model.pth")
    with open("rnn_vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def predict(model_path, vocab_path, input_strings):
    char_dim = 16
    hidden_dim = 64
    sentence_length = 10

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, hidden_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = []
    for s in input_strings:
        x_ids = [vocab.get(c, vocab["unk"]) for c in s]
        if len(x_ids) < sentence_length:
            x_ids += [0] * (sentence_length - len(x_ids))  # 补齐
        else:
            x_ids = x_ids[:sentence_length]  # 截断
        x.append(x_ids)

    with torch.no_grad():
        preds = model(torch.LongTensor(x))

    for i, s in enumerate(input_strings):
        print(f"输入：{s}, 预测'a'首次出现位置：{preds[i].item()}")


if __name__ == "__main__":
    # 训练模型
    main()

    # 测试预测
    test_strings = ["abcxyzlmno", "zzzaaaaaaa", "qwertyasdf", "bbbbaaaaaz"]
    predict("rnn_model.pth", "rnn_vocab.json", test_strings)
