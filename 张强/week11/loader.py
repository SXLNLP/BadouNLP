# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.data = []
        self.train_sample = config["train_sample"]
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        # self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                if isinstance(line, dict):
                    self._process_train_data(line)

    def _process_train_data(self, line):
        """处理训练数据并生成Prefix Mask"""
        question = line.get("title", "")
        answer = line.get("content", "")

        # 编码输入
        inputs = self.tokenizer(
            text=question,
            text_pair=answer,
            padding="max_length",
            max_length=self.config["max_length"],
            truncation=True,
            return_tensors="pt"
        )

        # 计算各部分长度
        question_tokens = self.tokenizer.tokenize(question)
        answer_tokens = self.tokenizer.tokenize(answer)
        question_len = len(question_tokens) + 2  # [CLS] + Q + [SEP]
        answer_len = len(answer_tokens) + 1  # A + [SEP]

        # 生成Prefix Mask（Question双向，Answer因果）
        seq_len = inputs.input_ids.shape[1]
        prefix_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)  # 下三角
        prefix_mask[:question_len, :question_len] = 1  # Question部分全连接

        # 结合Padding Mask
        final_mask = (inputs.attention_mask * prefix_mask)

        # 设置labels
        labels = inputs.input_ids.clone()
        labels[:, :question_len] = -100  # 遮盖Question
        if inputs.input_ids.shape[1] > question_len + answer_len:
            labels[:, question_len + answer_len:] = -100  # 遮盖Padding

        self.data.append({
            "input_ids": inputs.input_ids[0],
            "attention_mask": final_mask,  # 使用组合后的Mask
            "labels": labels[0]
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("sample_data.json", Config)
    print(dg[0])
    print(dg[0]["attention_mask"][19],dg[0]["attention_mask"][139])
    print(dg[0]["attention_mask"].shape)