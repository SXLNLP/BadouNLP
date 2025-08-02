# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载，适配BERT+mask自回归任务
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

    # 文本转index，头尾可加[CLS][SEP]
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

    # 补齐或截断
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id

    # 适配BERT自回归mask任务
    def prepare_data(self, title, content):
        # content编码：[CLS] content [SEP]
        content_ids = self.encode_sentence(content, self.config["input_max_length"], True, True)
        # title编码：title
        title_ids = self.encode_sentence(title, self.config["output_max_length"], False, False)
        # 拼接输入：[CLS] content [SEP] title
        input_ids = content_ids + title_ids
        input_ids = input_ids[:self.config["input_max_length"] + self.config["output_max_length"]]
        # mask掉title部分
        mask_token = self.vocab["[MASK]"]
        input_masked = content_ids + [mask_token] * len(title_ids)
        input_masked = input_masked[:len(input_ids)]
        # label：content部分为-100，title部分为真实token id
        label = [-100] * len(content_ids) + title_ids
        label = label[:len(input_ids)]
        # padding
        pad_len = self.config["input_max_length"] + self.config["output_max_length"] - len(input_ids)
        input_masked += [self.vocab["[PAD]"]] * pad_len
        label += [-100] * pad_len
        self.data.append([
            torch.LongTensor(input_masked),
            torch.LongTensor(label)
        ])
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

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_data_path"], Config, 1)
    for batch in dl:
        input_ids, labels = batch
        print(input_ids.shape, labels.shape)
        break

