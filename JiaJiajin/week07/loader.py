# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 二分类的标签映射，假设 0 和 1 代表两个类别
        self.index_to_label = {0: '类别0', 1: '类别1'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            try:
                self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
            except Exception as e:
                print(f"加载BertTokenizer失败: {e}")
                raise
        else:
            try:
                self.vocab = load_vocab(config["vocab_path"])
            except FileNotFoundError as e:
                print(f"加载词汇表失败: {e}")
                raise
            self.config["vocab_size"] = len(self.vocab)

        self.load()


    def load(self):
        self.data = []
        try:
            with open(self.path, 'r', encoding='utf8') as f:
                # 读取无表头的CSV文件，手动指定分隔符为逗号
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue  # 跳过格式错误的行

                    # 第一列为标签，第二列为文本
                    label_str = row[0].strip()
                    label = int(label_str)
                    if label not in self.label_to_index.values():
                        continue

                    text = row[1].strip()
                    if self.config["model_type"] == "bert":
                        input_id = self.tokenizer.encode(text,
                                                         max_length=self.config["max_length"],
                                                         padding='max_length',
                                                         truncation=True)
                    else:
                        input_id = self.encode_sentence(text)

                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([label])
                    self.data.append([input_id, label_index])
        except FileNotFoundError as e:
            print(f"文件 {self.path} 未找到: {e}")
            raise
        except ValueError as e:
            print(f"标签解析错误: {e}，请检查标签列是否为整数")
            raise
        return self.data


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
    dg = DataGenerator("H:/八斗网课/第七周 文本分类/week7 文本分类问题/文本分类练习.csv", Config)
    print(dg[1])
