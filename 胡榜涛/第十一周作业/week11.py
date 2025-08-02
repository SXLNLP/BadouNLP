# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json
"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        #self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.bert = BertModel.from_pretrained(r"D:\AI课程学习\bert-base-chinese" ,return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None,mask=None):
        mask_batch=[]
        if y is not None:
            for mask_index in mask:
                mask_index=mask_index.reshape(1,-1)
                mask_target = np.dot(mask_index.T,mask_index)
                for i in range(20,mask_target.shape[0]):
                    for j in range(mask_target.shape[1]):
                        if j>i:
                            mask_target[i,j] = 0
                        else:
                            mask_target[i,j] = 1
                mask_batch.append(mask_target)
            x, _ = self.bert(x, attention_mask=mask)
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
    corpus = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
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

def build_sample_and_mask(tokenizer, window_size, corpus):
    max_sentence_len = 20
    max_target_len = 100
    total_len = max_sentence_len + max_target_len
    line=random.choice(corpus)
    x = tokenizer(line["title"], add_special_tokens=False, padding="max_length", truncation=True,
                  max_length=total_len)
    y = tokenizer(line['content'], add_special_tokens=False, padding="max_length", truncation=True,
                  max_length=max_target_len+1)
    mask = x["attention_mask"]
    input_ids_x = x["input_ids"][:max_sentence_len]
    input_ids_y = y["input_ids"]
    input_ids_tmp = input_ids_x + input_ids_y
    input_ids =input_ids_tmp[:-1]
    target_ids=input_ids_tmp[1:]
    return input_ids, target_ids,mask






# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    dataset_mask = []
    for i in range(sample_length):
        x, y ,mask= build_sample_and_mask(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_mask.append(mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.LongTensor(dataset_mask)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过100字则终止迭代
        while pred_char != "\n" and len(openings) <= 100:
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

    pretrain_model_path = r"D:\AI课程学习\bert-base-chinese"
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
            x, y ,mask= build_dataset(batch_size, tokenizer, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer, window_size))
        print(generate_sentence("6月1日起北京实施“史上最严控烟期”", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"D:\AI课程学习\week10 文本生成问题\transformers-生成文章标题\sample_data.json", False)
