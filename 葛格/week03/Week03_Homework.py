# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
使用RNN完成多分类任务：
构造长度为8且包含字符'a'的字符串，
标签为字符'a'第一次出现的位置（0-7）。
"""

# RNN分类模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)              # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(x)          # h_n: (1, batch, hidden_size)
        h_last = h_n.squeeze(0)            # (batch, hidden_size)
        logits = self.classifier(h_last)   # (batch, num_classes)
        if y is not None:
            return self.loss_func(logits, y)
        else:
            return torch.softmax(logits, dim=1)

# 构建字符表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for idx, ch in enumerate(chars):
        vocab[ch] = idx + 1
    vocab["unk"] = len(vocab)
    return vocab

# 构造单个样本（字符串长度为8，且包含至少一个a）
def build_sample(vocab, sentence_length=8):
    while True:
        x_chars = [random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(sentence_length)]
        if 'a' in x_chars:
            break
    label = x_chars.index('a')  # 第一个 a 的位置作为标签
    x_ids = [vocab.get(ch, vocab["unk"]) for ch in x_chars]
    return x_ids, label

# 构建样本集
def build_dataset(sample_count, vocab, sentence_length=8):
    dataset_x, dataset_y = [], []
    for _ in range(sample_count):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 创建模型
def build_model(vocab_size, embedding_dim, hidden_size, num_classes):
    return RNNClassifier(vocab_size, embedding_dim, hidden_size, num_classes)

# 评估函数
def evaluate(model, vocab, sentence_length=8, num_classes=8):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    with torch.no_grad():
        y_pred = model(x)
        y_pred_label = torch.argmax(y_pred, dim=1)
        correct = (y_pred_label == y).sum().item()
    acc = correct / len(y)
    print(f"准确率：{acc:.4f}")
    return acc

# 主函数
def main():
    # 参数配置
    epoch_num = 10
    batch_size = 20
    train_sample = 1000
    embedding_dim = 16
    hidden_size = 32
    sentence_length = 8
    num_classes = sentence_length  # 8类：a在0~7位置
    learning_rate = 0.005

    vocab = build_vocab()
    model = build_model(len(vocab), embedding_dim, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for _ in range(train_sample // batch_size):
            x_batch, y_batch = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (train_sample // batch_size)
        print(f"第{epoch + 1}轮 平均Loss: {avg_loss:.4f}")
        evaluate(model, vocab, sentence_length)

    # 保存模型与词表
    torch.save(model.state_dict(), "rnn_classifier.pth")
    with open("rnn_vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

# 预测函数
def predict(model_path, vocab_path, input_strings):
    embedding_dim = 16
    hidden_size = 32
    sentence_length = 8
    num_classes = 8

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(len(vocab), embedding_dim, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = []
    for s in input_strings:
        ids = [vocab.get(ch, vocab["unk"]) for ch in s]
        if len(ids) < sentence_length:
            ids += [0] * (sentence_length - len(ids))
        else:
            ids = ids[:sentence_length]
        x.append(ids)
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        predicted = torch.argmax(result, dim=1)
    for s, p in zip(input_strings, predicted):
        print(f"输入: {s}, 'a'首次出现的位置是: {p.item()}")

if __name__ == "__main__":
    # main()
    test_strings = ["aabcdefg", "bcadefgh", "xyzaaaaa"]
    predict("rnn_classifier.pth", "rnn_vocab.json", test_strings)
