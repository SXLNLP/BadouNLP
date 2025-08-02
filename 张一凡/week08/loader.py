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
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
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

    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())

        # 确保每个类别至少有两个样本
        valid_categories = [k for k in standard_question_index if len(self.knwb[k]) >= 2]
        if not valid_categories:
            raise ValueError("每个类别至少需要两个样本用于训练")

        # 随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            p = random.choice(valid_categories)
            s1, s2 = random.sample(self.knwb[p], 2)

            # 选择负样本类别
            neg_categories = [k for k in standard_question_index if k != p and len(self.knwb[k]) > 0]
            if not neg_categories:
                return self.random_train_sample()

            n_question = random.choice(neg_categories)
            s3 = random.choice(self.knwb[n_question])
            return [s1, s2, s3]  # anchor, positive, negative

        # 随机负样本
        else:
            # 确保能选择两个不同类别
            if len(valid_categories) < 2:
                return self.random_train_sample()

            p, n = random.sample(valid_categories, 2)
            s1 = random.choice(self.knwb[p])
            s2 = random.choice(self.knwb[n])

            # 选择同类别下的另一个样本作为positive
            other_samples = [x for x in self.knwb[p] if not torch.equal(x, s1)]
            if not other_samples:
                return self.random_train_sample()

            s3 = random.choice(other_samples)
            return [s1, s3, s2]  # anchor, positive, negative


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator(Config["valid_data_path"], Config)
    print(dg[0])  # 测试三元组生成
