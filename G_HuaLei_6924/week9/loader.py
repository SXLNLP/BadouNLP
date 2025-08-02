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
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.config = config
        self.path = data_path
        # self.vocab = load_vocab(config["vocab_path"])
        # self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = [8] # 对句子使用 tokenizer对应预训练字标签时，句子开始标签
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                sent = "".join(sentenece)
                self.sentences.append(sent)
                input_ids = self.encode_sentence(sentenece)
                # input_ids = self.encode_sentence(sent)
                # labels.append(8) # 对句子使用 tokenizer对应预训练字标签时，句子结束标签
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        # input_id = []
        # if self.config["vocab_path"] == "words.txt":
        #     for word in jieba.cut(text):
        #         input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        # else:
        #     for char in text:
        #         input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # if padding:
        #     input_id = self.padding(input_id)
        # return input_id
        # return self.tokenizer(text,
        #                         padding="max_length",
        #                         max_length=self.config["max_length"],
        #                         truncation=True)
        return self.tokenizer.encode(text, add_special_tokens=True, max_length=self.config["max_length"], truncation=True, padding="max_length")

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
# def load_vocab(vocab_path):
#     from transformers import AutoTokenizer
#     # token_dict = {}
#     # with open(vocab_path, encoding="utf8") as f:
#     #     for index, line in enumerate(f):
#     #         token = line.strip()
#     #         token_dict[token] = index + 1  #0留给padding位置，所以从1开始
#     # return token_dict
#     return AutoTokenizer.from_pretrained("bert-base-chinese").get_vocab()

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    # dg = DataGenerator("../ner_data/train.txt", Config)

    tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])
    set1 = '我(1900)放到哈算法和撒旦法'
    set2 = list(set1)
    print(f'set1: {set1}')
    print(f'set2: {set2}')
    input_id1 = tokenizer.encode(set1, add_special_tokens=True, max_length=Config["max_length"], truncation=True, padding="max_length")
    input_id2 = tokenizer.encode(set2, add_special_tokens=True, max_length=Config["max_length"], truncation=True, padding="max_length")
    print(f'input_id1: {input_id1}')
    print(f'input_id2: {input_id2}')

