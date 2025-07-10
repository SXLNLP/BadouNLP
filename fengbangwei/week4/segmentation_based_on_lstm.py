# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/4 11:05
@Auth ： fengbangwei
@File ：segmentation_based_on_lstm.py.py

"""

import torch
import torch.nn as nn
import jieba
import numpy as np
from torch.utils.data import DataLoader

"""
作业：
实现基于词表，对于输入文本输出所有可能的分词方式。
尝试实现基于lstm的分词。
"""


class TorchModel(nn.Module):
    def __init__(self, char_dim, hidden_size, num_layers, vocab):
        super(TorchModel, self).__init__()
        # shape=(vocab_size, dim)  加1是为了预留一个特殊索引给 padding 索引（padding_idx）
        self.embedding = nn.Embedding(len(vocab) + 1, char_dim, padding_idx=0)
        self.lstm = nn.LSTM(char_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(hidden_size * 2, 2)  # 双向输出维度翻倍
        # 设置忽略标签值为 -100 的样本，使其不对损失计算产生影响。常用于处理变长序列中填充（padding）部分的标签
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        x = self.embedding(x)  # torch.Size([20, 20, 50])
        x, _ = self.lstm(x)  # torch.Size([20, 20, 100])
        y_pred = self.classify(x)
        # print(y.shape) #torch.Size([20, 20])
        # print(y_pred.shape) #torch.Size([20, 20, 2])
        if y is not None:
            # (batch_size*sen_len, 2) 将 20 x 20 x 2 变成 400 x 2    # 将 batch_size*sen_len 400 行
            return self.loss_func(y_pred.reshape(-1, 2), y.view(-1))
        else:
            return y_pred


# 构建词表
def build_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        # 使用 enumerate(f) 遍历文件中的每一行，同时获取行的索引（从0开始）和内容。index 表示当前行号，line 是当前行的字符串内容。
        for index, line in enumerate(f):
            vocab[line.strip()] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab


# 构建数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = DataSet(corpus_path, vocab, max_length)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return data_loader


# 数据集
class DataSet:
    def __init__(self, corpus_path, vocab, max_length):
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.max_length = max_length
        self.data = []
        self.load()

    def load(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                sequence = sentence_to_sequence(sentence, self.vocab)
                label = sequence_to_label(sentence)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                # 使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 20000:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label


# 文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence


# 基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    # print(len(label))
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label


#  训练
def main():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每批训练条数
    char_dim = 50  # 字符维度
    max_length = 20  # 每条文本的最大长度
    hidden_size = 100  # 隐藏层维度
    learning_rate = 1e-3  # 学习率
    num_layers = 1  # RNN的层数
    vocab_path = "chars.txt"
    corpus_path = "corpus.txt"
    vocab = build_vocab(vocab_path)
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)
    model = TorchModel(char_dim, hidden_size, num_layers, vocab)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x_batch, y_batch in data_loader:
            optim.zero_grad()  # 梯度清零
            loss = model(x_batch, y_batch)
            loss.backward()  # 反向传播
            optim.step()  # 更新参数
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model_lstm.pth")


# 预测
def predict(model_path, input_strings):
    char_dim = 50  # 字符维度
    hidden_size = 100  # 隐藏层维度
    num_layers = 1  # RNN的层数
    vocab_path = "chars.txt"
    vocab = build_vocab(vocab_path)
    model = TorchModel(char_dim, hidden_size, num_layers, vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重  防止潜在的反序列化攻击)
    model.eval()
    for input_string in input_strings:
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            # 第一个是模型输出
            # 第二个可能是隐藏状态
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  # 预测出的01序列    dim=-1 1 20 2 取2列的那个维度
            # 在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()


if __name__ == '__main__':
    # main()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]

    predict("model_lstm.pth", input_strings)