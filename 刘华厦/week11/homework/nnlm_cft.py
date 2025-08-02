# coding:utf8
import json

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


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, q, x, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))

            # 1. 找到每个样本中SEP标记的位置（问题和答案的分界点）
            sep_token_id = 102  # bert默认是102
            sep_positions = []
            for i in range(x.shape[0]):
                # 找到第一个SEP的位置（问题结束的位置）
                sep_idx = (x[i] == sep_token_id).nonzero(as_tuple=True)[0]
                if len(sep_idx) == 0:
                    # 如果没有SEP，默认整个序列都是问题（不mask）
                    sep_positions.append(x.shape[1])
                else:
                    sep_positions.append(sep_idx[0].item() + 1)  # 答案开始的位置
            # 3. 对问题部分解除mask（让问题内部可以互相关注）
            for i in range(x.shape[0]):
                answer_start = sep_positions[i]
                # 问题部分：[0, answer_start)，设置为全1（允许互相关注）
                mask[i, :answer_start, :answer_start] = 1.0

            # 4. 处理padding部分（全部mask）
            pad_mask = (x == 0).unsqueeze(1)  # (batch_size, 1, seq_len)
            base_mask = mask.masked_fill(pad_mask, 0.0)  # padding位置禁止关注
            # print(mask, mask.shape)
            if torch.cuda.is_available():
                base_mask = base_mask.cuda()
            x, _ = self.bert(x, attention_mask=base_mask,
                output_attentions=False,
                output_hidden_states=False)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


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
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            corpus += title + "<sep>" + content
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    title = corpus.split("<sep>")[0]
    content = corpus.split("<sep>")[1]
    end = random.randint(1, len(content))
    window = title + "[sep]" + content[0:end]
    target = content[0:end + 1]  # 输入输出错开一位

    q = tokenizer.encode(title, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return q, x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_q = []
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        q, x, y = build_sample(tokenizer, window_size, corpus)
        dataset_q.append(q)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_q), torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(char_dim, vocab, pretrain_model_path)
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


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'D:\work\个人资料\NLP算法\week6 语言模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            q, x, y = build_dataset(batch_size, tokenizer, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(q, x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"./sample_data.json", False)