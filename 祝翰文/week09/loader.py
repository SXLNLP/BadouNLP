# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载 - 适配BERT
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.label2id = {label: idx for idx, label in enumerate(self.schema)}
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                tokens = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    tokens.append(char)
                    labels.append(label)

                # 使用BERT tokenizer处理
                tokenized = self.tokenizer(
                    tokens,
                    is_split_into_words=True,  # 表示输入已分词
                    padding="max_length",
                    truncation=True,
                    max_length=self.config["max_length"],
                    return_tensors="pt"
                )

                # 对齐标签
                word_ids = tokenized.word_ids()
                aligned_labels = []
                previous_word_idx = None
                for word_idx in word_ids:
                    # 特殊标记 [CLS], [SEP], [PAD] 设置为 -1
                    if word_idx is None:
                        aligned_labels.append(-1)
                    # 同一个token的不同部分
                    elif word_idx == previous_word_idx:
                        aligned_labels.append(-1)
                    else:
                        aligned_labels.append(self.label2id[labels[word_idx]])
                    previous_word_idx = word_idx

                self.data.append({
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                    "labels": torch.LongTensor(aligned_labels)
                })
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
        # 确保标签包含"O"
        if "O" not in schema:
            schema.append("O")
        return schema


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    Config["bert_path"] = "bert-base-chinese"
    dg = DataGenerator("../ner_data/train.txt", Config)
