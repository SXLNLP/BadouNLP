# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel

"""
基于pytorch的LSTM语言模型
"""


class SFTModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path, freeze_bert=False):
        super(SFTModel, self).__init__()
        # 载入预训练模型BERT
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        # 冻结BERT的参数，进行SFT时，只微调线性层
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 分类层
        self.classify = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None):
        # BERT输出
        x, _ = self.bert(x)

        # 分类层输出
        y_pred = self.classify(x)

        if y is not None:
            # 计算loss（监督学习）
            return self.compute_loss(y_pred, y)
        else:
            # 预测时返回概率分布
            return torch.softmax(y_pred, dim=-1)

    def compute_loss(self, y_pred, y):
        # 交叉熵损失
        return nn.CrossEntropyLoss()(y_pred.view(-1, y_pred.size(-1)), y.view(-1))


# 加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True,
                         max_length=10)  # 将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = SFTModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings


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


# 初始化模型
def build_model(vocab_size, hidden_size, pretrain_model_path):
    model = SFTModel(hidden_size, vocab_size, pretrain_model_path)
    return model


# 训练模型
def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 128
    train_sample = 10000
    char_dim = 768
    window_size = 10
    vocab_size = 21128
    learning_rate = 0.001

    pretrain_model_path = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)