# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config):
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.data = self.load(data_path)

    def load(self, path):
        dataset = []
        with open(path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence, label_list = [], []
                for line in segment.strip().split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    label_list.append(self.schema[label])
                text = "".join(sentence)
                encoding = self.tokenizer(text,
                                          max_length=self.config["max_length"],
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors="pt")
                label_list = label_list[:self.config["max_length"]]
                label_list += [-1] * (self.config["max_length"] - len(label_list))
                dataset.append([
                    encoding["input_ids"].squeeze(0),
                    encoding["attention_mask"].squeeze(0),
                    torch.LongTensor(label_list)
                ])
        return dataset

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

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
    dg = DataGenerator("../ner_data/train.txt", Config)

