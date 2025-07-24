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
from collections import defaultdict

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        if config["use_bert"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        else:
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = [8]
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                input_ids = self.encode_sentence(sentenece)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        if self.config["use_bert"]:
            # 使用BERT的tokenizer
            encoded = self.tokenizer.encode(
                text,
                padding="max_length" ,
                max_length=self.config["max_length"],
                truncation=True
            )
            # return encoded["input_ids"].squeeze(0)  # 返回token ids
            return encoded # 返回token ids
        else:
            # 原有的编码逻辑
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
    def decode(self, sentence, labels):
        sentence = "$" + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)+2]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            print("location", s,e)
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            print("org", s,e)
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            print("per", s,e)
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            print("time", s,e)
            results["TIME"].append(sentence[s:e])
        return results

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]  # 截断
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))  # 填充
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

