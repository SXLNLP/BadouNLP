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
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")  # 两次换行分开，是因为数据问题
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()  # 一个字对应一个label，一个字对应一个label
                    sentence.append(char)
                    labels.append(self.schema[label])  # 转化成类别对应的编号
                self.sentences.append("".join(sentence))
                # todo
                if self.config["model_type"] == "bert":
                    input_ids = self.tokenizer(
                        sentence,  # 已分词的 list
                        is_split_into_words=True,  # 告诉 tokenizer 输入是已经分词的 list
                        padding="max_length",  # 可选：补全到最大长度
                        truncation=True,  # 可选：截断超长句子
                        max_length=self.config["max_length"],
                        return_tensors="pt"   # 返回 PyTorch 张量
                    )["input_ids"]
                    input_ids = input_ids.squeeze(0)  # 去掉 batch_size维度
                else:
                    input_ids = self.encode_sentence(sentence)  # 编码 ===>（一个字符为单位）
                labels = self.padding(labels, -1)
                self.data.append([input_ids, torch.LongTensor(labels)])
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

    def encode_token_list(self, token_list):
        input_id = []
        for char in token_list:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
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

