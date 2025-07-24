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
        self.index_to_label = {0: '差评', 1: '好评'}
            #, 2: '股票', 3: '社会', 4: '文化',
            #                  5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
            #                   10: '体育', 11: '科技', 12: '汽车', 13: '健康',
            #                  14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f.readlines():
                line = line.split(",")
                tag = line[0]
                #label = self.label_to_index[tag]
                label=int(tag)
                title = line[1].strip()
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
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
    dg = DataGenerator(data_path, config) # 列表[张量input,张量Y]
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../h_data/train.csv", Config) # 样本
    print(dg[1])
    dl = load_data("../h_data/train.csv",Config) # 可迭代对象，转化成batch后的

    print(f"总样本数: {len(dg)}")
    print(f"总批次数: {len(dl)}")
    # 遍历 DataLoader 并打印每个批次的形状
    for batch_idx, batch in enumerate(dl):
        print(f"Batch {batch_idx + 1}:")
        print(f"  input_ids shape: {batch[0].shape}")  # 输入张量形状 (batch_size, max_length)
        print(f"  labels shape: {batch[1].shape}")  # 标签张量形状 (batch_size,)
        print("-" * 30)
        break  # 仅演示第一个批次

