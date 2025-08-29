# -*- coding: utf-8 -*-

import json
import re
import os
import torch
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
        self.label_to_index = {label: i for i, label in enumerate(config["ner_tags"])}
        self.config["class_num"] = len(self.label_to_index)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                text = line["text"]
                labels = line["labels"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], padding="max_length", truncation=True)
                    label_ids = self.encode_labels(labels, input_id)
                else:
                    input_id = self.encode_sentence(text)
                    label_ids = self.encode_labels(labels, input_id)
                input_id = torch.LongTensor(input_id)
                label_ids = torch.LongTensor(label_ids)
                self.data.append([input_id, label_ids])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def encode_labels(self, labels, input_id):
        label_ids = [self.label_to_index[label] for label in labels]
        label_ids = self.padding(label_ids)
        return label_ids

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_ner.json", Config)
    print(dg[1])
