# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                #加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            if self.config["loss_type"] ==  "TripletLoss":
                return self.random_train_tripletloss() #随机生成一个训练样本 [s1, s2, s3]
            else:
                return self.random_train_sample()  # 随机生成一个训练样本 [s1, s2, label]
        else:
            return self.data[index]

    #依照一定概率生成负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        #随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            p = random.choice(standard_question_index)
            #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(self.knwb[p], 2)
                return [s1, s2, torch.LongTensor([1])]
        #随机负样本
        else:
            #从FAQ里面任意取两组索引
            p, n = random.sample(standard_question_index, 2)
            #分别从两组FAQ里面任取一个问题
            s1 = random.choice(self.knwb[p])
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([-1])]

    """ 【作业新写方法】随机取两个同一组问题，再取1个不同组问题，组成TripletLoss需要用的3组向量 """
    #从所有FAQ样本中司机取出三个问题对应编码，对应TripletLoss(a,p,n)中的a,p,n
    def random_train_tripletloss(self):
        #所有问题的索引
        all_question_index = list(self.knwb.keys())
        # 使用推导式，从字典中选取value个数大于等于2的字典索引，用于TripletLoss(a,p,n)中的a、p
        same_question_index = list(key for key, value in self.knwb.items() if len(value) > 1)
        #从same_question_index中任取两组索引
        a, p = random.sample(same_question_index, 2)
        #从all_question_index中踢掉a, p后，得到的新所有问题的索引
        new_all_question_index = list(key for key in all_question_index if key not in [a, p])
        #再从new_all_question_index中任取一个索引，用于TripletLoss(a,p,n)中的n，这里的n已经不可能跟a、p重复了，上面已经过滤了
        n =random.sample(new_all_question_index, 1)[0]
        #从a,p,n索引中取出对应的句子编码
        s_a = random.choice(self.knwb[a])
        s_p = random.choice(self.knwb[p])
        s_n = random.choice(self.knwb[n])
        return  [s_a, s_p, s_n]


#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
