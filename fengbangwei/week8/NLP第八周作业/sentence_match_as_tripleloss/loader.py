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
        self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        # 初始化一个默认值为空列表的字典，用于存储知识库 {"0": [标准问1, 标准问2, ...], "1": [标准问1, 标准问2, ...], ...}
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集 {"questions": ["改下畅聊套餐"], "target": "畅聊套餐变更"}
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集 ["话费", "话费查询"] question label
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

    # 补齐或截断输入的序列，使其可以在一个batch内运算
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
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 依照一定概率生成负样本或正样本
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        # 将知识库的键（问题索引）转换为列表，以便于后续处理  0 1 2 3 4
        standard_question_index = list(self.knwb.keys())
        rp, rn = random.sample(standard_question_index, 2)
        # 确保 rp 类别下至少有两个样本
        while len(self.knwb[rp]) < 2 and rp != rn:
            rp = random.choice(list(self.knwb.keys()))
        # a：anchor 原点
        # p：positive 与a同一类别的样本
        # n：negative 与a不同类别的样本
        a, p = random.sample(self.knwb[rp], 2)
        n = random.choice(self.knwb[rn])
        return [a, p, n]


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator(r"D:\AI\ai_project\deepseek\week8\data\train.json", Config)
    dg.data
    print(dg[1])  # 调用getitem方法
    print(len(dg))  # 200,  # 每轮训练中采样数量
    print(random.random())
