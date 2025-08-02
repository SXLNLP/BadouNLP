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
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = self.load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        self.load()

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1
        return token_dict

    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf8") as f:
            return json.loads(f.read())

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
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
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
        return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_triplet_sample()
        return self.data[index]

    def random_triplet_sample(self):
        standard_question_index = list(self.knwb.keys())
        while True:
            p, n = random.sample(standard_question_index, 2)
            if len(self.knwb[p]) >= 2:
                a, pos = random.sample(self.knwb[p], 2)
                neg = random.choice(self.knwb[n])
                return [a, pos, neg]

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
