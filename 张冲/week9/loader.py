# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config['pretrain_model_path'])
        self.vacob = load_vocab(config['vocab_path'])
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

                sentenece_encode = self.tokenizer("".join(sentenece), return_tensors="pt", truncation=True,
                                                  max_length=self.config['max_length'], padding="max_length")
                i = 0
                label_encode = []
                input_ids = sentenece_encode.data['input_ids'][0]
                for input_id, word_id in zip(input_ids, sentenece_encode.word_ids()):
                    if word_id is None:
                        label_encode.append(-1)
                    else:
                        label_encode.append(labels[i])
                        if sentenece[i:i + len(self.vacob[input_id.item()])] == self.vacob[input_id.item()]:
                            i += len(self.vacob[input_id.item()])
                self.sentences.append("".join(sentenece))
                label_encode = self.padding(labels, -1)
                self.data.append([input_ids, torch.LongTensor(label_encode)])
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

    # 补齐或截断输入的序列，使其可以在一个batch内运算
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


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[index] = token
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator(Config['train_data_path'], Config)
