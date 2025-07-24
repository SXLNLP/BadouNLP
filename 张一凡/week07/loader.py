# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.label_map = config["label_map"]
        self.config["class_num"] = len(self.label_map)

        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.data = self.load_data()

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置
        return token_dict

    def load_data(self):
        data = []
        df = pd.read_csv(self.path)

        for _, row in df.iterrows():
            # 使用配置中的列名
            label = row[self.config["label_column"]]
            text = row[self.config["text_column"]]

            if self.config["model_type"] == "bert":
                inputs = self.tokenizer.encode_plus(
                    text,
                    max_length=self.config["max_length"],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = inputs['input_ids'].squeeze(0)
            else:
                input_ids = self.encode_sentence(text)

            label_index = torch.LongTensor([label])  # 直接使用数值label
            data.append((input_ids, label_index))
        return data

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab.get("[UNK]", 1)))
        input_id = self.padding(input_id)
        return torch.LongTensor(input_id)

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, shuffle=True):
    dataset = DataGenerator(data_path, config)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=2
    )
    return loader
