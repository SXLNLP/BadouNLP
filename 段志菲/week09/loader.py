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
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.config["vocab_size"] = len(self.tokenizer)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                # 使用BERT tokenizer编码
                encoded = self.tokenizer(
                    sentenece,
                    max_length=self.config["max_length"],
                    truncation=True,
                    padding="max_length",
                    is_split_into_words=True,
                    return_tensors="pt"
                )
                input_ids = encoded["input_ids"].squeeze(0)
                # 处理labels的alignment
                word_ids = encoded.word_ids()
                aligned_labels = []
                current_word = None
                for word_id in word_ids:
                    if word_id is None:
                        # 特殊token
                        aligned_labels.append(-1)
                    elif word_id != current_word:
                        # 新词开始
                        aligned_labels.append(labels[word_id])
                        current_word = word_id
                    else:
                        # 同一个词的后续部分，使用I-标签或保持原标签
                        label = labels[word_id]
                        if label % 2 == 1:  # 如果是B-标签
                            label += 1  # 转为I-标签
                        aligned_labels.append(label)
                
                labels = torch.LongTensor(aligned_labels)
                self.data.append([input_ids, labels])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)
