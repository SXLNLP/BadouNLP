# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

"""
基于pytorch的BERT+mask自回归语言模型
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.input_dim = input_dim
        self.vocab_size = len(vocab)

        self.embedding = nn.Embedding(len(vocab), input_dim)
        self.pos_encoding = PositionalEncoding(input_dim)

        # 使用Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,  # 多头注意力头数
            dim_feedforward=input_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def create_causal_mask(self, seq_len):
        """创建掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x, y=None):
        batch_size, seq_len = x.shape

        # 词嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.input_dim)  # 缩放嵌入
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)  # 位置编码

        # 创建掩码
        causal_mask = self.create_causal_mask(seq_len)
        if torch.cuda.is_available():
            causal_mask = causal_mask.cuda()

        # Transformer编码
        x = self.transformer(x, mask=causal_mask)

        # 分类层
        y_pred = self.classify(x)  # output shape:(batch_size, seq_len, vocab_size)

        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 60:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings


def sampling_strategy(prob_distribution, temperature=1.0, top_k=50, top_p=0.9):
    """
    减少重复生成
    """
    prob_distribution = prob_distribution / temperature
    prob_distribution = torch.softmax(prob_distribution, dim=-1)

    # Top-k采样：只考虑概率最高的k个token
    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(prob_distribution, min(top_k, prob_distribution.size(-1)))
        # 创建mask，只保留top-k的概率
        prob_distribution = torch.zeros_like(prob_distribution)
        prob_distribution.scatter_(-1, top_k_indices, top_k_probs)

    # Top-p采样：只考虑累积概率达到p的token
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(prob_distribution, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过的token
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        # 将不需要的token概率设为0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        prob_distribution[indices_to_remove] = 0

    # 归一化
    prob_distribution = prob_distribution / prob_distribution.sum()

    # 随机采样
    prob_distribution = prob_distribution.cpu().numpy()
    return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20  #训练轮数
    batch_size = 32  #每次训练样本个数
    train_sample = 50000  #每轮训练总共训练的样本总数
    char_dim = 512  #每个字的维度 = 8*64
    window_size = 10  #样本文本长度
    vocab = build_vocab("vocab.txt")  #建立字表
    corpus = load_corpus(corpus_path)  #加载语料
    model = build_model(vocab, char_dim)  #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  #学习率
    print("BERT+mask语言模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
