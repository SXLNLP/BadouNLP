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
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__))+"/../../models/bert-base-chinese")
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
                if(self.config["model"] != "bert"):
                    input_ids = self.encode_sentence(sentenece)
                else:
                    if(self.config["bert_tokenizer"] == "origin"):
                        #bert的tokenizer在会自动进行分词，无法实现每个输入字与标注对应，所以尽量不要用这个原生方法
                        input_ids = self.tokenizer.encode_plus(sentenece,
                                                        max_length=self.config["max_length"],
                                                        pad_to_max_length=True,
                                                        add_special_tokens=True)
                        #tokenizer返回的是字典(dict-like)，需要取出里面的list
                        input_ids = input_ids["input_ids"]
                    elif(self.config["bert_tokenizer"] == "custom"):
                        #bert中字表中没有的字符，需要用self.tokenizer.vocab.get('[UNK]')替代
                        input_ids = [self.tokenizer.vocab.get(char, self.tokenizer.vocab.get('[UNK]')) for char in sentenece]
                        #bert 中的补齐字符索引，不能用-1，而是用self.tokenizer.pad_token_id
                        input_ids = self.padding(input_ids, self.tokenizer.pad_token_id)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
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
    dg = DataGenerator("../ner_data/train.txt", Config)

