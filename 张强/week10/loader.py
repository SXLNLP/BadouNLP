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
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.data = []
        self.train_sample = config["train_sample"]
        self.window_size = config["window_size"]
        # self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        corpus=''
        with open(self.path, encoding="gbk") as f:
            for line in f:
                corpus += line.strip()

        # 确保有足够的数据
        if len(corpus) < self.window_size + 1:
            raise ValueError("语料库太小，无法生成样本")

        # 生成多个样本
        for i in range(self.train_sample):
            start = random.randint(0, len(corpus) - 1 - self.window_size)
            end = start + self.window_size
            window = corpus[start:end]
            target = corpus[start + 1:end + 1]

            # 编码
            x = self.encode_sentence(window)
            y = self.encode_sentence(target)

            # 创建attention mask（下三角矩阵）
            ar_mask = torch.tril(torch.ones(self.window_size, self.window_size))

            # 填充到max_length
            x = self.padding(x, self.tokenizer.pad_token_id)
            y = self.padding(y, self.tokenizer.pad_token_id)

            self.data.append([
                torch.LongTensor(x),
                ar_mask,
                torch.LongTensor(y)
            ])

    def encode_sentence(self, text, padding=True):
        # 不再需要 max_length 参数，因为外部已经通过 window_size 限制长度
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if padding:
            input_ids = self.padding(input_ids, self.tokenizer.pad_token_id)
        return input_ids

    # def padding_attention_mask(self, mask, original_len):
    #     """填充attention mask到max_length"""
    #     padded_mask = torch.zeros((self.config["max_length"], self.config["max_length"]),dtype=torch.long)
    #     padded_mask[:original_len, :original_len] = mask
    #     return padded_mask

    def padding(self, input_id, pad_token=0):
        # 直接填充到 window_size 的长度
        input_id = input_id[:self.window_size]  # 安全截断（理论上不应发生）
        input_id += [pad_token] * (self.window_size - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]




# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("corpus.txt", Config)
    dl = DataLoader(dg, batch_size=32)
    for x,attention_mask, y in dl:
        print(x.shape,attention_mask.shape, y.shape)
        print(x, attention_mask, y)