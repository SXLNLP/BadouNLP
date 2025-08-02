# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/7/21
# @Author      : liuboyuan
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
        # News classification label mapping
        LABELS = ['文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '证券', '农业', '电竞', '民生']
        self.index_to_label = {idx: name for idx, name in enumerate(LABELS)}
        self.label_to_index = {name: idx for idx, name in self.index_to_label.items()}
        self.schema = self.label_to_index
        print(self.schema)
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.load()


    def load(self):
        """
        Load dataset with each line being a JSON object:
        {"text": "...", "label_name": "...", "label": int}
        """
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                title = sample["text"]
                # support both integer label id and label name
                if "label" in sample:
                    label = int(sample["label"])
                else:
                    label_name = sample.get("label_name")
                    label = self.label_to_index.get(label_name, -1)
                if label not in self.index_to_label:
                    # skip unknown label
                    continue
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(
                        title,
                        max_length=self.config["max_length"],
                        truncation=True,
                        padding='max_length'
                    )
                else:
                    input_id = self.encode_sentence(title)
                self.sentences.append(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
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
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])