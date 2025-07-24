# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd

"""
数据加载
"""


# 重写DataGenerator类处理csv数据
class DataGenerator:
    def __init__(self, data_path, config, mode="train"):
        self.config = config
        self.mode = mode
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}  # 二分类标签
        self.label_to_index = {v: k for k, v in self.index_to_label.items()}

        # 从csv加载数据
        self.df = pd.read_csv(data_path, encoding='gb18030')
        # 统计正负样本
        pos_count = len(self.df[self.df.label == 1])
        neg_count = len(self.df[self.df.label == 0])
        print(f"正样本数: {pos_count}, 负样本数: {neg_count}")

        # 划分训练集/验证集
        if mode == "train":
            self.data = self.df.sample(frac=0.8, random_state=config["seed"])
        else:
            self.data = self.df.drop(self.df.sample(frac=0.8, random_state=config["seed"]).index)

        if config["model_type"].startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row["review"]
        label = int(row["label"])

        if self.config["model_type"].startswith("bert"):
            inputs = self.tokenizer(
                text,
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].squeeze(0)
        else:
            input_ids = self.encode_sentence(text)

        return input_ids, torch.LongTensor([label])

    def encode_sentence(self, text):
        input_id = [self.vocab.get(char, self.vocab["[UNK]"]) for char in text]
        input_id = self.padding(input_id)
        return torch.LongTensor(input_id)

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        return input_id + [0] * (self.config["max_length"] - len(input_id))


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


# if __name__ == "__main__":
#     from config import Config
#
#     dg = DataGenerator("valid_tag_news.json", Config)
#     print(dg[1])
