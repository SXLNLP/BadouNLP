#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel, BertConfig

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-chinese", config=config)

        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.classify = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.encoder(x)[0]
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
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
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    # x = tokenizer.encode(window, return_tensors="pt", padding='max_length', truncation=True, max_length=window_size+2)
    # y = tokenizer.encode(target, return_tensors="pt", padding='max_length', truncation=True, max_length=window_size+2)
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', max_length=window_size)   #将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', max_length=window_size)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    # return x[0], y[0]
    return x, y

#建立数据集
#batch_size 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(batch_size, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(batch_size):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    # return dataset_x, dataset_y
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
    # return torch.stack(dataset_x), torch.stack(dataset_y)

#建立模型
# def build_model():
#     # model = LanguageModel(char_dim, vocab)
#     # return model
#     return LanguageModel(char_dim)

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    reverse_vocab = dict((v,k) for k,v in tokenizer.get_vocab().items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过20字则终止迭代
        # 原始一维张量
        cls_token = torch.tensor([101])
        sep_token = torch.tensor([102])
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = list(openings[-window_size:])
            x = tokenizer.convert_tokens_to_ids(x)
            # x = torch.tensor(tokenizer.convert_tokens_to_ids(x))
            # x = torch.cat([cls_token, x, sep_token], dim=0)
            print(f'x: {x}')
            x = torch.LongTensor([x])
            # x = torch.stack([x])
            if torch.cuda.is_available():
                x = x.cuda()
            # y = model(x)[0][-1]
            y = model(x)
            y = y[0][-1]
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


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    # char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    # vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    # model = build_model(vocab, char_dim)    #建立模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    config = BertConfig.from_pretrained("bert-base-chinese", num_hidden_layers=2)
    print('config = ', config)
    model = LanguageModel(config)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
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
    train("corpus.txt", False)