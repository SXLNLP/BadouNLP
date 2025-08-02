# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载 - 修改为适配BERT
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用BERT的tokenizer替代原来的vocab
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        # 添加数字处理相关配置
        self.split_numbers = config.get("split_numbers", True)  # 是否将数字拆分为单个字符
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.label2id = {v: k for k, v in self.schema.items()}  # 反转schema
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    # 处理数字字符
                    if self.split_numbers and char.isdigit():
                        sentence.extend(list(char))  # 将数字拆分为单个字符
                        labels.extend([label] * len(char))
                    else:
                        sentence.append(char)
                        labels.append(label)
                text = "".join(sentence)
                self.sentences.append(text)

                # 使用BERT tokenizer处理文本
                encoded = self.tokenizer(
                    text,
                    max_length=self.config["max_length"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    return_offsets_mapping=True
                )

                # 获取原始字符到token的映射
                char_to_token = []
                for i in range(len(text)):
                    char_to_token.append(encoded.char_to_token(i))

                # 处理标签对齐
                label_ids = []
                for token_idx in range(len(encoded.input_ids[0])):
                    if token_idx == 0 or token_idx == len(encoded.input_ids[0]) - 1:  # [CLS]或[SEP]
                        label_ids.append(-100)
                        continue

                    # 获取当前token对应的原始字符位置
                    token_start, token_end = encoded.token_to_chars(token_idx)
                    if token_start is None:  # 特殊token
                        label_ids.append(-100)
                        continue

                    # 如果token对应多个字符，取第一个字符的标签
                    first_char_pos = token_start
                    if first_char_pos < len(labels):
                        label = labels[first_char_pos]
                        # 如果是B-标签且token对应多个字符，后面的字符应该用I-标签
                        if token_end - token_start > 1 and label % 2 == 1:  # B-标签
                            label += 1  # 转为I-标签
                        label_ids.append(label)
                    else:
                        label_ids.append(-100)

                self.data.append({
                    "input_ids": encoded["input_ids"].squeeze(0),  # 去掉batch维度
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": torch.LongTensor(label_ids)
                })
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)
