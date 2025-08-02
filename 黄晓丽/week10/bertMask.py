#coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""
使用Bert+mask做自回归语言模型训练
"""


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=6, nhead=8):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(self, x, y=None):
        # 创建注意力掩码
        src_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # 嵌入层 + 位置编码
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        # Transformer编码器
        x = self.transformer(x, mask=src_mask)

        # 分类层
        logits = self.classifier(x)

        if y is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                y.view(-1),
                ignore_index=0  # 忽略padding
            )
            return loss
        else:
            return F.softmax(logits, dim=-1)

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（上三角矩阵）"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0, "<UNK>": 1}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 2 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]

    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim=256):
    return TransformerLanguageModel(len(vocab), embed_dim=char_dim)

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = {v: k for k, v in vocab.items()}
    model.eval()
    with torch.no_grad():
        current = openings
        # 生成了换行符，或生成文本超过30字则终止迭代
        while not current.endswith("\n") and len(current) <= 30:
            # 准备输入序列
            input_seq = current[-window_size:]
            x = [vocab.get(char, vocab["<UNK>"]) for char in input_seq]

            # 填充到固定长度
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x

            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()

            # 预测下一个字符的概率分布
            probs = model(x)[0][-1]  # 只取最后一个位置的预测

            # 采样策略
            index = sampling_strategy(probs)
            next_char = reverse_vocab.get(index, "<UNK>")
            current += next_char

    return current

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:  # 90%贪心，10%采样
        return int(torch.argmax(prob_distribution))
    else:
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(len(prob_distribution), p=prob_distribution)

def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    char_dim = 256        #每个字的维度
    window_size = 20       #样本文本长度
    vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料

    model = build_model(vocab)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)   #建立优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    print("Transformer语言模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optim.step()
            watch_loss.append(loss.item())

        scheduler.step()
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
    return model



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
