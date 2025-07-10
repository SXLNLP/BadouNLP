# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""
class DataGenerator:
    def __init__(self, data_df, config):
        self.config = config
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        else:
            self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab) if not self.config[
                                                               "model_type"] == "bert" else self.tokenizer.vocab_size

        self.data_df = data_df  # 保存传入的DataFrame
        self.data = self.load()  # 调用load方法处理数据


    def load(self):
        data = []
        for index, row in self.data_df.iterrows():
            review = row['review']
            label = row['label']
            label_index = int(label)
            # label_index = self.label_to_index[label]

            if self.config["model_type"] == "bert":
                # 使用BERT的分词器
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], padding='max_length',
                                                 truncation=True)
            else:
                # 使用旧的词表分词
                input_id = self.encode_sentence(review)

            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label_index])
            data.append([input_id, label_index])
        return data

    def encode_sentence(self, text):
        input_id = []
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
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

def load_and_split_data(config, shuffle=True):
    df = pd.read_csv(config["data_path"], encoding='utf-8')

    # 切分数据集
    train_df, valid_df = train_test_split(
        df,
        test_size=config["test_split_ratio"],
        random_state=config["seed"],
        stratify=df['label']  # 保持训练集和验证集中标签比例一致
    )

    # 为训练集和验证集分别创建DataGenerator和DataLoader
    train_dg = DataGenerator(train_df, config)
    valid_dg = DataGenerator(valid_df, config)

    train_dl = DataLoader(train_dg, batch_size=config["batch_size"], shuffle=shuffle)
    valid_dl = DataLoader(valid_dg, batch_size=config["batch_size"], shuffle=False)

    return train_dl, valid_dl

# if __name__ == "__main__":
#     from config import Config
#     dg = DataGenerator("valid_tag_news.json", Config)
#     print(dg[1])
