# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from model import LanguageModel
from config import Config
"""
数据加载
"""

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer,window_size,corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end=start + window_size
    window=corpus[start:end]
    target=corpus[start+1:end+1]#输入输出错开一位
    
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   #将字转换成序号
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    
    return x,y

#建立数据集
def build_dataset(sample_length,tokenizer,window_size,corpus):
    dataset_x=[]#输入
    dataset_y=[]#输出
    for i in range(sample_length):
        x,y=build_sample(tokenizer,window_size,corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

def load_corpus(path):
    corpus = " "
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#构建模型
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

