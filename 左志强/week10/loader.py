# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    #文本到对应的index
    #头尾分别加入[cls]和[sep]
    def encode_sentence(self, text, max_length, with_cls_token=True, with_sep_token=True):
        input_id = []
        if with_cls_token:
            input_id.append(self.vocab["[CLS]"])
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if with_sep_token:
            input_id.append(self.vocab["[SEP]"])
        input_id = self.padding(input_id, max_length)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id

    #输入输出转化成序列
    def prepare_data(self, title, content):
        input_seq = self.encode_sentence(content, self.config["input_max_length"], False, False) #输入序列
        output_seq = self.encode_sentence(title, self.config["output_max_length"], True, False) #输出序列

        gold = self.encode_sentence(title, self.config["output_max_length"], False, True) #不进入模型，用于计算loss

        self.data.append([torch.LongTensor(input_seq),
                          torch.LongTensor(output_seq),
                          torch.LongTensor(gold)])

        return


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_data_path"], Config, 1)
    print(dl[1])

