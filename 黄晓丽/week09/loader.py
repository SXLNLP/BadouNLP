# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 替换为BERT的tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], use_fast=True)
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.data = []
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

                # BERT编码处理
                encoded = self.tokenizer.encode_plus(
                    text=sentenece,
                    max_length=self.config["max_length"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    is_split_into_words=True
                )

                # 对齐标签（处理subword问题）
                word_ids = encoded.word_ids()
                aligned_labels = []
                prev_word = None
                for word_id in word_ids:
                    if word_id is None:  # 特殊token [CLS]/[SEP]
                        aligned_labels.append(-100)
                    elif word_id != prev_word:  # 当前词的首个子词
                        aligned_labels.append(labels[word_id])
                    else:  # 后续子词
                        aligned_labels.append(-100)  # 忽略
                    prev_word = word_id
                self.data.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": torch.LongTensor(aligned_labels)
                })
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
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

#加载字表或词表
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



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)

