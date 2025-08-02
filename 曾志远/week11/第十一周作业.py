#coding:utf8
import json
import logging

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
from transformers import BertTokenizer

"""
基于pytorch的bert语言模型  stf训练
"""

logger = logging.getLogger(__name__)

Config = {}


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size, pretrained_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_path, attn_implementation="eager", return_dict=False)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 生成图因果掩码 attention_mask
            # [[1, 1, 1, 1, 1, 0, 0, 0]
            #  [1, 1, 1, 1, 1, 0, 0, 0]
            #  [1, 1, 1, 1, 1, 0, 0, 0]
            #  [1, 1, 1, 1, 1, 0, 0, 0]
            #  [1, 1, 1, 1, 1, 0, 0, 0]
            #  [1, 1, 1, 1, 1, 1, 0, 0]
            #  [1, 1, 1, 1, 1, 1, 1, 0]
            #  [1, 1, 1, 1, 1, 1, 1, 1]]
            # 矩阵宽度 w = (first_seq_length + 1) + (second_seq_length + 1)
            # 上半部分
            top_half = torch.concat((torch.ones((x.shape[0], Config['first_seq_length'], Config['first_seq_length'])), (
                torch.zeros((x.shape[0], Config['first_seq_length'], Config['second_seq_length'])))), dim=-1)
            bottom_half = torch.concat(
                (torch.ones((x.shape[0], Config['second_seq_length'], Config['first_seq_length'])),
                 torch.tril(torch.ones((x.shape[0], Config['second_seq_length'], Config['second_seq_length'])))),
                dim=-1)
            att_mask = torch.concat((top_half, bottom_half), dim=1)
            if torch.cuda.is_available():
                att_mask = att_mask.cuda()
            x = self.bert(x, attention_mask=att_mask)[0]  # (batch_size,sen_len,input_dim)
            y_pred = self.classify(x)  #output shape:(batch_size, sen_len, vocab_size)
            y_pred = self.dropout(y_pred)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x = self.bert(x)[0]  # (batch_size,sen_len,input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)
            y_pred = self.dropout(y_pred)
            return torch.softmax(y_pred, dim=-1)


# title 结论  content 新闻
# 加载所有数据  输入： content + [CLS] + title  输出： title + [SEP]
# seq -> seq
class DataGenerator:
    def __init__(self, data_path, tokenizer):
        self.first_seq_length = None
        self.second_seq_length = None
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.load()

    def load(self):
        self.data = []
        self.first_seq_length = 0
        self.second_seq_length = 0
        self.pre_data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = json.loads(line)
                title = line['title']
                content = line['content']
                if len(title) > self.second_seq_length:
                    Config['second_seq_length'] = self.second_seq_length = len(title) + 1  # +1是给[SEP]留位置
                if len(content) > self.first_seq_length:
                    Config['first_seq_length'] = self.first_seq_length = len(content) + 1  # +1是给[CLS]留位置
                self.pre_data.append((title, content))
        for title, content in self.pre_data:
            self.prepare_data(title, content)
        return

    def prepare_data(self, title, content):                  #   first_seq_length                   second_seq_length
        input_ids = self.encode_sentence(title, content)     #   s1         +    [CLS]        +      s2    +      [SEP]
        output_ids = self.encode_sentence(title, '')  # [-100,-100,...] + s2[0]        +  s2[1:]+[SEP] +  [-100]
        self.data.append([torch.LongTensor(input_ids), torch.LongTensor(output_ids)])

    def encode_sentence(self, title: str, content: str):
        if content:  # 如果是输入数据
            first_seq = self.tokenizer.encode(content + self.cls_token, add_special_tokens=False, padding='max_length',
                                              max_length=self.first_seq_length)
            second_seq = self.tokenizer.encode(title, add_special_tokens=False, max_length=self.second_seq_length,
                                               padding='max_length')
            input_ids = first_seq + second_seq
        else:
            first_label = [-100] * (self.first_seq_length-1)
            second_label = self.tokenizer.encode(title + self.sep_token, add_special_tokens=False,
                                                 padding='max_length',
                                                 max_length=self.second_seq_length+1)
            second_label = [-100 if x == 0 else x for x in second_label]
            input_ids = first_label + second_label
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 加载数据
def load_data(data_path, batch_size, tokenizer, shuffle=True):
    dg = DataGenerator(data_path, tokenizer)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl


#建立模型
def build_model(vocab_size, char_dim, pretrained_path):
    model = LanguageModel(char_dim, vocab_size, pretrained_path)
    return model


#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        openings = openings + '[CLS]'
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "[SEP]" and len(openings) <= 200:
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


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    pretrained_path = r"F:\八斗ai课程\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    vocab_size = len(tokenizer)
    epoch_num = 100  #训练轮数
    batch_size = 32  #每次训练样本个数
    char_dim = 768  #每个字的维度
    train_data = load_data(corpus_path, batch_size, tokenizer, shuffle=True)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    model = build_model(vocab_size, char_dim, pretrained_path)  # 建立模型
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            x, y = batch_data  #构建一组训练样本
            optim.zero_grad()  #梯度归零
            loss = model(x, y)  #计算loss
            loss.backward()  #计算梯度
            optim.step()  #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence(
            "阿根廷布宜诺斯艾利斯省奇尔梅斯市一服装店，8个月内被抢了三次。最后被抢劫的经历，更是直接让老板心理崩溃：歹徒在抢完不久后发现衣服“抢错了尺码”，理直气壮地拿着衣服到店里换，老板又不敢声张，只好忍气吞声。",
            model, tokenizer))
        print(generate_sentence(
            "今天（27日）上午10:43左右，正在张掖举行的首届丝绸之路（张掖）国际通用航空大会上，沈阳飞行家表演队的一架XA42飞机在表演特技动作时发生坠机，目前人员伤亡情况不明，消防车和救护车已赶至现场。",
            model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("sample_data.json", False)
