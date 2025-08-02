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
数据加载 - 适配BERT版本
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用BERT tokenizer代替自定义vocab
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_name"])
        self.schema = self.load_schema(config["schema_path"])
        self.label2id = {v: k for k, v in self.schema.items()}  # 反转标签映射
        self.sentences = []
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
                    sentence.append(char)
                    labels.append(self.schema[label])
                
                # 将字符列表转换为字符串
                text = "".join(sentence)
                self.sentences.append(text)
                
                # 使用BERT tokenizer处理文本
                encoding = self.tokenizer(
                    text,
                    max_length=self.config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_offsets_mapping=True
                )
                
                # 获取分词后的input_ids和attention_mask
                input_ids = encoding["input_ids"].squeeze(0)
                attention_mask = encoding["attention_mask"].squeeze(0)
                
                # 处理标签对齐问题
                bert_labels = self.align_labels(text, labels, encoding["offset_mapping"].squeeze(0))
                
                self.data.append([
                    input_ids,
                    attention_mask,
                    torch.LongTensor(bert_labels)
                ])
        return

    def align_labels(self, text, original_labels, offset_mapping):
        """
        将原始标签与BERT分词后的token对齐
        """
        # 初始化标签序列，全部填充为-100（表示忽略）
        labels = [-100] * len(offset_mapping)
        
        # 遍历所有token的偏移映射
        for i, (start, end) in enumerate(offset_mapping):
            # 跳过特殊token ([CLS], [SEP], [PAD])
            if start == 0 and end == 0:
                continue
                
            # 对于完整字符的token，直接使用原始标签
            if end - start == 1:
                char_index = start
                if char_index < len(original_labels):
                    labels[i] = original_labels[char_index]
            # 对于子词token，使用原始标签或特殊处理
            else:
                # 获取当前token对应的文本
                token_text = text[start:end]
                
                # 如果是中文子词（如##词），使用前一个token的标签
                if token_text.startswith("##"):
                    if i > 0 and labels[i-1] != -100:
                        labels[i] = labels[i-1]
                # 否则使用第一个字符的标签
                elif start < len(original_labels):
                    labels[i] = original_labels[start]
        
        # 确保标签序列长度与input_ids一致
        assert len(labels) == len(offset_mapping)
        return labels

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
    dl = DataLoader(
        dg, 
        batch_size=config["batch_size"], 
        shuffle=shuffle,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch])
        )
    )
    return dl
