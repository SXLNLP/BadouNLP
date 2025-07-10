# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer  # 生成Bert的词表对应的数字序列（embedding前三个数字序列），包括cls和sep这两个token
from sklearn.model_selection import train_test_split as tts


"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.load()

    """
    1、加载文件中的数据，获取内容对应的索引
    2、Bert模型使用tokenizer处理成数字序列
    3、将数字序列和对应的索引张量处理
    """

    def load(self):
        self.data = []
        df = pd.read_csv(self.path)
        reviews = df['review'].tolist()
        labels = df['label'].tolist()
        self.config["class_num"] = len(set(labels))
        for review, label in zip(reviews, labels):
            if self.config["model_type"] == "bert":
                input_x = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_x = self.encode_sentence(review)
            self.sentences.append(review)
            input_x = torch.LongTensor(input_x)
            label = torch.LongTensor([label])
            self.data.append((input_x, label))
        return

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
            token = line.strip()  # ❓
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

# 划分训练数据集和验证数据集
def split_data(data_path, config):
    df = pd.read_csv(data_path)
    train_df, val_df = tts(df, test_size=0.2, random_state=42)
    train_df.to_csv(config["train_data_path"], index=False)  # 保存训练集
    val_df.to_csv(config["valid_data_path"], index=False)

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)  # ❓只为了打乱吗？
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("/Users/juju/nlp20/class7/hwAndPrac/hw/data/train_data.csv", Config)
    print(dg[1])
