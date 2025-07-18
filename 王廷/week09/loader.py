# -*- coding: utf-8 -*-
import json
import re
import os
import torch
import random
import jieba
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
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))

                # BERT编码
                sentence_str = "".join(sentence)
                encoded = self.tokenizer.encode_plus(
                    sentence_str,
                    max_length=self.config["max_length"],
                    truncation=True,
                    padding="max_length",
                    return_tensors=None,
                    return_attention_mask=True
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                # 标签对齐
                bert_labels = []
                for word, label_id in zip(sentence, labels):
                    tokens = self.tokenizer.tokenize(word)
                    bert_labels.append(label_id)
                    bert_labels.extend([-1] * (len(tokens) - 1))  # 子词部分用-1填充

                # 截断标签
                bert_labels = bert_labels[:self.config["max_length"] - 2]  # 保留空间给[CLS]和[SEP]
                bert_labels = [-1] + bert_labels + [-1]  # 添加[CLS]和[SEP]位置
                bert_labels = self.padding(bert_labels, -1)  # 填充到max_length

                self.data.append([
                    torch.LongTensor(input_ids),
                    torch.LongTensor(bert_labels),
                    torch.LongTensor(attention_mask)
                ])

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl