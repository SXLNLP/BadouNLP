# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
# import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import csv

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.index_to_label = {0: '差评', 1: '好评'}  # 根据需要更改标签
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.index_to_label = {0: '差评', 1: '好评'}
        # self.label_to_index = {0: 0, 1: 1}  # 直接将数字标签映射到自身
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as csvf:
            csv_reader = csv.reader(csvf)
            next(csv_reader)
            for row in csv_reader:
                label = int(row[0])
                title = row[1]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

# 补齐或者截断输入的序列，使之能够在一个batch里面运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    # dg = DataGenerator(Config["valid_data_path"], Config)
    dg = DataGenerator(Config["train_data_path"], Config)
    print(dg[1])