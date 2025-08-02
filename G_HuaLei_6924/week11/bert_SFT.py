# coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence

"""
基于pytorch的LSTM语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=-1)
            # return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 108:
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

def train(corpus_path, save_weight=False):
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    # max_length = 128      # 传入 bert 的样本id 最大长度
    max_length = 50  # 传入 bert 的样本id 最大长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'E:\AI_study\八斗学院\录播课件\第六周\bert-base-chinese'
    # pretrain_model_path = 'bert-base-chinese'
    train_data_path = r'E:\AI_study\八斗学院\录播课件\第十周\week10 文本生成问题\week11_参考作业答案\week11作业\sample_data.json'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # corpus = load_corpus(corpus_path)     #加载语料
    train_data = load_data(tokenizer, train_data_path, batch_size, max_length)  # 加载训练集
    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        # print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

"""
数据加载
"""

class DataGenerator(Dataset):
    def __init__(self, data_path, bertTonkenizer, max_length):
        self.tokenizer = bertTonkenizer
        self.max_length = max_length
        self.path = data_path
        self.load()

    def load(self):
        # 初始化数据列表
        self.data = []
        # 打开文件
        with open(self.path, encoding="utf8") as f:
            # 遍历文件中的每一行
            for line in f:
                # 将每一行转换为json格式
                line = json.loads(line)
                # 如果行长度为0，则跳过
                if len(line) == 0:
                    continue
                # 加载训练集
                if isinstance(line, dict):
                    # 获取问题
                    question = line["title"]
                    # 获取标签
                    label = line["content"]
                    # 使用tokenizer对问题和标签进行编码
                    input_ids = self.tokenizer.encode(question, add_special_tokens=False, truncation=True,
                                                      max_length=self.max_length)
                    label_ids = self.tokenizer.encode(label, add_special_tokens=False, truncation=True,
                                                      max_length=self.max_length)
                    x = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id] + label_ids + [
                        self.tokenizer.sep_token_id]
                    y = [-1] * len(input_ids) + [-1] + label_ids + [self.tokenizer.sep_token_id]

                    mask = create_mask(len(input_ids), len(label_ids))
                    mask = pad_mask(mask, self.max_length)

                    x = x[: self.max_length] + [0] * (self.max_length - len(x))
                    y = y[: self.max_length] + [0] * (self.max_length - len(y))

                    # 将编码后的数据添加到数据列表中
                    self.data.append((torch.LongTensor(x), mask, torch.LongTensor(y)))
                else:
                    # 如果数据格式错误，则抛出异常
                    raise Exception("数据格式错误")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def create_mask(q_len, a_len):
    len_q = q_len + 2  # cls + sep
    len_a = a_len + 1  # sep
    mask = torch.ones(len_q + len_a, len_q + len_a)
    # 遍历 q_sequence 每个token
    for i in range(len_q):
        # q_sequence 对于 a_sequence 中每个token之后的内容都不知情
        mask[i, len_q:] = 0
    for j in range(len_a):
        # a_sequence 对于 a_sequence 中每个当前token之后的内容都不知情
        mask[len_q + j, len_q + j + 1:] = 0
    return mask

def pad_mask(tensor, target_shape_size):
    height_size, _ = tensor.shape
    # target_height, target_width = target_shape
    # 初始化一个全零张量，形状与实际输入bert 模型的数据保持一致，为输出传入 bert的mask做准备
    init_mask = torch.zeros((target_shape_size, target_shape_size), dtype=tensor.dtype, device=tensor.device)
    output_shape_size = min(height_size, target_shape_size)
    # 将原始mask张量的内容填入新 mask 当中
    init_mask[0:output_shape_size, 0:output_shape_size] = tensor[:output_shape_size, :output_shape_size]
    return init_mask


def load_data(tokenizer, train_data_path, batch_size, max_length):
    train_data = DataGenerator(train_data_path, tokenizer, max_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
