# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.data = []
        self.load()

    def load(self):
        df = pd.read_csv(self.path, encoding='utf8')
        for index, row in df.iterrows():
            # 按列名访问数据（推荐）
            label = row['label']
            review = row['review']
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],
                                                 pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id


class CustomDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    train_data, valid_data = train_test_split(
        dg.data,
        test_size=config['test_size'],
        random_state=config['seed'],
        shuffle=True
    )
    train_data_set = DataLoader(CustomDataSet(train_data), batch_size=config["batch_size"], shuffle=shuffle)
    valid_data_set = DataLoader(CustomDataSet(valid_data), batch_size=config["batch_size"], shuffle=shuffle)
    return train_data_set, valid_data_set


if __name__ == "__main__":
    from config import Config

    load_data(Config['data_path'], Config)
