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


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    questions = line["questions"]
                    label = line["target"]
                    label_index = torch.LongTensor([self.schema[label]])
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.data.append([input_id, label_index])
                else:
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
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

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


class TripletDataset(Dataset):
    def __init__(self, anchor_texts, positive_texts, negative_texts, tokenizer, max_len=128):
        self.anchors = anchor_texts
        self.positives = positive_texts
        self.negatives = negative_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = self.tokenizer(self.anchors[idx], padding='max_length', truncation=True,
                                max_length=self.max_len, return_tensors='pt')
        positive = self.tokenizer(self.positives[idx], padding='max_length', truncation=True,
                                  max_length=self.max_len, return_tensors='pt')
        negative = self.tokenizer(self.negatives[idx], padding='max_length', truncation=True,
                                  max_length=self.max_len, return_tensors='pt')

        return {
            'anchor_input_ids': anchor['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor['attention_mask'].squeeze(0),
            'positive_input_ids': positive['input_ids'].squeeze(0),
            'positive_attention_mask': positive['attention_mask'].squeeze(0),
            'negative_input_ids': negative['input_ids'].squeeze(0),
            'negative_attention_mask': negative['attention_mask'].squeeze(0),
        }



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
