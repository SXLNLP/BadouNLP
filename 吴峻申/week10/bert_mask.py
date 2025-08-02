# coding:utf8
# bert_mask.py
import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

"""
基于PyTorch和BERT的掩码语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # 创建BERT配置
        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=input_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
            type_vocab_size=1
        )
        # 初始化BERT模型
        self.bert = BertModel(config)
        # 分类层 (MLM头部)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        attention_mask = torch.ones_like(x)

        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state

        # 分类层预测
        y_pred = self.classify(sequence_output)

        if y is not None:
            # 计算掩码语言模型损失 (仅计算被掩码位置的损失)
            return self.loss(
                y_pred.view(-1, y_pred.shape[-1]),
                y.view(-1),
                ignore_index=-100  # 忽略非掩码位置
            )
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表 (添加[MASK]特殊标记)
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<UNK>": 1, "<MASK>": 2}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个MLM样本
def build_sample(vocab, window_size, corpus):
    # 随机选择文本片段
    start = random.randint(0, len(corpus) - 1 - window_size)
    window = corpus[start:start + window_size]

    # 初始输入和目标
    x = [vocab.get(char, vocab["<UNK>"]) for char in window]
    y = [-100] * window_size  # 初始化为忽略值

    # 确定要掩码的位置 (15%的token)
    mask_indices = random.sample(
        range(window_size),
        max(1, int(window_size * 0.15))  # 至少掩码一个token
    )

    for idx in mask_indices:
        original_char = window[idx]
        rand = random.random()

        # 80%概率替换为[MASK]
        if rand < 0.8:
            x[idx] = vocab["<MASK>"]
        # 10%概率替换为随机字符
        elif rand < 0.9:
            x[idx] = random.choice(list(vocab.values()))
        # 10%概率保持不变

        # 设置目标为原始字符
        y[idx] = vocab.get(original_char, vocab["<UNK>"])

    return x, y


# 建立数据集
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
    return LanguageModel(char_dim, vocab)


# 文本生成测试 (MLM不适合自回归生成，此函数仅供参考)
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = {v: k for k, v in vocab.items()}
    model.eval()
    with torch.no_grad():
        # 在开头添加[MASK]用于预测
        input_str = openings + "[MASK]" * 5
        x = [vocab.get(char, vocab["<UNK>"]) for char in input_str[-window_size:]]
        x = torch.LongTensor([x])

        if torch.cuda.is_available():
            x = x.cuda()

        y_pred = model(x)
        predictions = torch.argmax(y_pred, dim=-1).squeeze(0).cpu().numpy()

        # 只替换[MASK]位置
        result = ""
        for i, char in enumerate(input_str[-window_size:]):
            if char == "[MASK]":
                result += reverse_vocab.get(predictions[i], "<UNK>")
            else:
                result += char

    return openings + result


# 训练函数
def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
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
