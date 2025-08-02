#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer,BertModel
import json

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese", attn_implementation='eager')
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):

           #output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            mask_title = torch.ones((x.shape[0], 30, 30))
            mask[:, :30, :30] = mask_title
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            y_pred = y_pred[:,30:,:]
            return self.loss(y_pred.reshape(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

def bert_build_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path, truncation_side="right")


#加载语料


def load_corpus(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                # 确保只包含title和content字段
                filtered_data = {
                    'title': data.get('title', '').strip(),
                    'content': data.get('content', '').strip()
                }
                result.append(filtered_data)
            except json.JSONDecodeError:
                print(f"警告：跳过无效JSON行: {line}")
    return result

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def bert_build_sample(tokenizer, corpus):
    text = random.choice(corpus)
    title = text["title"]
    content = text["content"]
    x1 = tokenizer.encode(title, padding='max_length', truncation=True, max_length=30)
    x2 = tokenizer.encode(content, padding='max_length', truncation=True, max_length=111)
    x = x1 + x2[1:]
    y = tokenizer.encode(content, padding='max_length', truncation=True, max_length=110)
    return x, y




#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = bert_build_sample(tokenizer, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

def bert_build_model():
    model = LanguageModel()
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    reverse_vocab = dict((y, x) for x, y in tokenizer.get_vocab().items())

    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 300:
            openings += pred_char
            # x = [tokenizer.get_vocab().get(char, tokenizer.get_vocab()["[UNK]"]) for char in openings[-window_size:]]
            # if len(openings) > 30:
            #     openings = openings[1:]

            x = tokenizer.encode(openings)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
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
    epoch_num = 10        #训练轮数
    batch_size = 50       #每次训练样本个数
    train_sample = 500   #每轮训练总共训练的样本总数
    # char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    # vocab = build_vocab("vocab.txt")       #建立字表
    tokenizer = bert_build_vocab(r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese")
    corpus = load_corpus(corpus_path)     #加载语料
    # model = build_model(vocab, char_dim)    #建立模型
    model = bert_build_model()
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("世界最大胸罩将拍卖面积堪比两个网球场", model, tokenizer, window_size))
        # print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)

    # corpus = load_corpus('sample_data.json')
    # sum_title = 0
    # sum_content = 0
    # for x in corpus:
    #     print(len(x["title"]),len(x["content"]))
    #     sum_title += len(x["title"])
    #     sum_content += len(x["content"])
    # print(sum_title,sum_content)
    #
    # print(f"共加载 {len(corpus)} 条数据")
    # print("第一条数据 title:", corpus[0]["title"])
    # print("第一条数据 content:", corpus[0]["content"])
    #
    # tokenizer = bert_build_vocab(r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese")
    # x, y = build_dataset(1, tokenizer, corpus)
    # print(tokenizer.decode(x[0]))
    # print(tokenizer.decode(y[0]))

