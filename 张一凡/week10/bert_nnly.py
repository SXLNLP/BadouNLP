# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertConfig

"""
基于BERT+Mask的自回归语言模型
"""


class BertLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=6, num_heads=8):
        super(BertLanguageModel, self).__init__()
        # 自定义BERT配置
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
            is_decoder=True,  # 设置为解码器
            add_cross_attention=False
        )
        self.bert = BertModel(config)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, attention_mask=None, y=None):
        # BERT前向传播
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        y_pred = self.classify(sequence_output)

        if y is not None:
            # 计算损失时忽略padding部分
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = y_pred.view(-1, y_pred.shape[-1])[active_loss]
                active_labels = y.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
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
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    # 构建输入和标签
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]

    # 创建attention mask (1表示真实token，0表示padding)
    attention_mask = [1] * len(x)

    return x, y, attention_mask


# 建立数据集 (修改为返回attention mask)
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    attention_masks = []
    for i in range(sample_length):
        x, y, mask = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_masks.append(mask)
    return (
        torch.LongTensor(dataset_x),
        torch.LongTensor(dataset_y),
        torch.LongTensor(attention_masks)
    )


# 文本生成函数 
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            # 创建attention mask
            attention_mask = torch.ones_like(x)
            y = model(x, attention_mask=attention_mask)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings


# 采样策略 
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 训练函数
def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    window_size = 10  # 样本文本长度
    hidden_size = 256  # BERT隐藏层大小
    num_layers = 6  # BERT层数
    num_heads = 8  # 注意力头数

    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = BertLanguageModel(len(vocab), hidden_size, num_layers, num_heads)  # 建立BERT模型

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  # 使用更小的学习率

    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, attention_mask = build_dataset(batch_size, vocab, window_size, corpus)
            if torch.cuda.is_available():
                x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()

            optim.zero_grad()
            loss = model(x, attention_mask=attention_mask, y=y)
            loss.backward()
            optim.step()
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
    train("corpus.txt", False)
