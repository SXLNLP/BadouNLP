#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json
from torch.nn.utils.rnn import pad_sequence

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

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x,split_pos=None, y=None):
        if y is not None:
            batch, seq_len = x.shape
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device))
            # sep_pos = (x == tokenizer.sep_token_id).int().argmax(dim=-1)
            mask = causal_mask.unsqueeze(0).repeat(batch, 1, 1)
            for b, pos in enumerate(split_pos):
                mask[b, :pos, pos:] = 0  # title 不能看后面 content
            # print(mask, mask.shape)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料
def load_corpus(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, data):
    """
    随机选一条样本，把title当x，把content当y
    """
    item = random.choice(data)

    title_ids = tokenizer.encode(item["title"], add_special_tokens=False)   #将字转换成序号
    content_ids = tokenizer.encode(item["content"], add_special_tokens=False)
    input_ids = [tokenizer.cls_token_id] + title_ids + [tokenizer.sep_token_id] + content_ids + [tokenizer.sep_token_id]
    label_ids = [-100] * (len(title_ids) + 2) + content_ids + [tokenizer.sep_token_id]

    return torch.LongTensor(input_ids), torch.LongTensor(label_ids)

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, data):
    dataset_x = []
    dataset_y = []
    split_pos = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, data)
        dataset_x.append(x)
        dataset_y.append(y)

    sep_idx = (x == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
    content_start = sep_idx[0] + 1 if len(sep_idx) > 0 else 0
    split_pos.append(content_start)

    dataset_x = pad_sequence(dataset_x, batch_first=True, padding_value=tokenizer.pad_token_id)
    dataset_y = pad_sequence(dataset_y, batch_first=True, padding_value=-100)
    split_pos = torch.LongTensor(split_pos)
    return dataset_x, dataset_y, split_pos

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
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
    epoch_num = 20        #训练轮数
    batch_size = 16       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    

    pretrain_model_path = r'D:\八斗课程视频\第六周 语言模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, split_pos = build_dataset(batch_size, tokenizer, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, split_pos=split_pos,y=y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
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
    train(r"D:\八斗课程视频\第十周\week10 文本生成问题\transformers-生成文章标题/sample_data.json", False)
