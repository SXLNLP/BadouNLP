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


class DataGenerator(Dataset):  # 修改：继承Dataset类
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []  # 修改：修正变量名拼写错误
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])

                # 修改：使用tokenizer的新参数格式，显式指定padding和truncation
                encoding = self.tokenizer(
                    sentence,
                    is_split_into_words=True,
                    max_length=self.config["max_length"],
                    padding="max_length",  # 修改：使用padding参数替代pad_to_max_length
                    truncation=True,  # 修改：显式启用截断
                    return_tensors="pt"
                )

                # 修改：正确处理标签维度，保持与input_ids一致
                labels = self.padding(labels, -100)  # 修改：使用-100作为标签填充值（与CrossEntropyLoss兼容）
                input_id = encoding["input_ids"].squeeze(0)  # 修改：直接从encoding获取并调整维度
                attention_mask = encoding["attention_mask"].squeeze(0)  # 新增：添加attention_mask

                # 修改：调整标签维度，确保为1D张量 [seq_len]
                label_index = torch.LongTensor(labels)

                # 修改：添加attention_mask并调整数据结构
                self.data.append({
                    "input_ids": input_id,
                    "attention_mask": attention_mask,
                    "labels": label_index
                })

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]  # 修改：返回字典格式，包含所有必要字段

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)

    # 修改：添加collate_fn处理批次数据
    def collate_fn(batch):
        # 提取各字段
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # 使用pad_sequence动态填充
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle, collate_fn=collate_fn)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("ner_data/train", Config)
    # 打印第一个样本的形状信息，用于调试
    if len(dg) > 0:
        sample = dg[0]
        print(f"样本 input_ids 形状: {sample['input_ids'].shape}")
        print(f"样本 labels 形状: {sample['labels'].shape}")
        print(f"样本 attention_mask 形状: {sample['attention_mask'].shape}")
