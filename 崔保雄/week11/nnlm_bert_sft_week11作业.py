#coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
"""
利用bert进行SFT微调，实现回答类的生产式任务。
需要使用seq-to-seq形式的mask
"""

class LanguageModel(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        """ 使用bert """
        self.layer = BertModel.from_pretrained(bert_path, return_dict=False)
        """ 线形层的输入维度，使用bert的hidden_size """
        hidden_size = self.layer.config.hidden_size
        self.classify = nn.Linear(hidden_size, self.layer.config.vocab_size)

        # self.dropout = nn.Dropout(0.1)#0.1
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):

        # y_pred = self.dropout(y_pred)
        if y is not None:
            """ 这里的mask使用的是seq-to-seq mask，用于回答类的生成式模型 """
            x, _ = self.layer(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)

            y_pred_2 = y_pred.view(-1, y_pred.shape[-1])
            y_2 = y.view(-1)
            return self.loss(y_pred_2, y_2)
        else:
            x, _ = self.layer(x)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)
            return torch.softmax(y_pred, dim=-1) #dim可以不赋值，默认就是-1，代表使用最后一个维度

#加载语料, 用title当成假想的prompt，content当成假想的answer
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        # 构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = create_mask(len(prompt_encode), len(answer_encode))
        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

""" 使用张量拼接的方式，生成seq-to-seq形式的mask """
def create_mask(seq_x_len, seq_y_len):
    # 左上角，prompt之间互相可见
    mask_top_left = torch.ones(seq_x_len, seq_x_len)
    # 右上角，answer对prompt不可见
    mask_top_right = torch.zeros(seq_x_len, seq_y_len)

    # 左下角，prompt对answer可见
    mask_bottom_left = torch.ones(seq_y_len, seq_x_len)
    # 右下角，answer对answer部分可见，后面的文本能看到前面的。这里先初始化弄个全0，后续要继续弄成下三角互相可见
    mask_bottom_right = torch.ones(seq_y_len, seq_y_len)
    # 右下角的继续弄成下三角互相可见
    mask_bottom_right = torch.tril(mask_bottom_right, diagonal=0)

    # 上面的两个，横向拼接
    mask_top = torch.cat((mask_top_left, mask_top_right), dim=1)
    # 下面的两个，也横向拼接
    mask_bottom = torch.cat((mask_bottom_left, mask_bottom_right), dim=1)

    # 将前面先横向拼接的两个，再进行纵向拼接
    mask = torch.cat((mask_top, mask_bottom), dim=0)
    # print("mask.shape：\n", mask.shape)
    # print("mask：\n", mask)
    return mask

def pad_mask(mask, target_shape):
    #原mask张量的形状，height行数，width列数
    height, width = mask.shape
    #最终mask张量的形状，target_h行数，target_w列数
    target_h, target_w = target_shape

    mask_target = torch.zeros(target_w, target_h)
    h_end = min(height, target_h)
    w_end = min(width, target_w)
    mask_target[:h_end, :w_end] = mask[:h_end, :w_end]
    return mask_target

#建立模型
def build_model(bert_path):
    model = LanguageModel(bert_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过50字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        #贪婪解码，取最大概率的一个选中
        strategy = "greedy"
    else:
        # 随机采样，概率越大的，选中几率最大
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        # p参数： 1D 数组，可选。表示每个元素被选中的概率。该数组的长度应与 a 的长度相同，且所有概率之和应为 1。如果没有指定 p，则所有元素的概率相同。
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def train(corpus_path, bert_path, save_weight=True):
    epoch_num = 20        #训练轮数 20
    batch_size = 32       #每次训练样本个数 64
    train_sample = 10000   #每轮训练总共训练的样本总数 50000
    # char_dim = 768  # 每个字的维度
    max_length = 50  # 样本文本长度 50
    # char_dim = 256        #每个字的维度
    # window_size = 10       #样本文本长度
    # vocab = build_vocab(bert_vocab_path)       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    # vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率

    model = build_model(bert_path)    #建立模型

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  # 建立数据集

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))
    if not save_weight:
        return
    else:
        # base_name = os.path.basename(corpus_path).replace("txt", "pth")
        # model_path = os.path.join("model", base_name)
        model_path = os.path.join("output", "nnlm_bert_sft.pth")
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    bert_path = r"D:\www.root\bert-base-chinese"
    train("sample_data.json", bert_path)
