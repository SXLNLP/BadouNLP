# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel
from config import Config
from loader import load_data

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):

    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.bert = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False,
                                              num_hidden_layers=2)
        input_dim = self.bert.config.hidden_size
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # x = self.embedding(x)  # output shape:(batch_size, sen_len, input_dim) 64 10 256
        seq_len = x.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).bool()

        x = self.bert(x, attention_mask=causal_mask)[0]  # output shape:(batch_size, sen_len, input_dim) 64 10 256
        x = self.dropout(x)
        y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size) 64 10 3961
        if y is not None:
            # y_pred.shape  # (2, 3, 5)   -> 每个时间步输出词汇表的概率分布
            # y.shape  # (2, 3)      -> 真实标签（每个词是一个索引）
            # y_pred.view(-1, y_pred.shape[-1])  # shape: (6, 5)        640 3961
            # y.view(-1)  # shape: (6,)                                 640
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    sampling_strategy.history_tokens = []  # 每次生成前清空历史
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings


def sampling_strategy(logits, repetition_penalty=1.2, temperature=0.7, top_k=50, top_p=0.9):
    # 清除缓存状态（generate_sentence 中调用）
    if not hasattr(sampling_strategy, "history_tokens"):
        sampling_strategy.history_tokens = []

    # 应用重复惩罚
    for token_id in set(sampling_strategy.history_tokens):
        logits[token_id] /= repetition_penalty

    # softmax + 温度缩放
    probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()

    # Top-k 采样
    indices = np.argsort(probs)[-top_k:]
    top_probs = probs[indices]

    # Nucleus (Top-p) 采样
    sorted_indices = np.argsort(top_probs)[::-1]
    sorted_probs = top_probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    sorted_probs[cumulative_probs > top_p] = 0
    sorted_probs /= sorted_probs.sum()  # 归一化

    # 随机选择
    next_token_idx = np.random.choice(len(sorted_probs), p=sorted_probs)
    next_token = indices[sorted_indices[next_token_idx]]

    # 记录已生成 token
    sampling_strategy.history_tokens.append(next_token)

    return int(next_token)


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
    epoch_num = 1000  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 512  # 每个字的维度
    window_size = 64  # 样本文本长度
    vocab = build_vocab("vocab.txt")  # 建立字表
    train_data = load_data(corpus_path, Config, window_size, shuffle=True)
    model = build_model(vocab, char_dim)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=3e-5)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            x, y = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
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


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
