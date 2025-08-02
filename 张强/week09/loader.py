# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.max_length = config["max_length"]
        # self.vocab = load_vocab(config["vocab_path"])
        # self.config["vocab_size"] = len(self.vocab)
        # 初始化 BERT tokenizer - 禁用特殊 token
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.label_map = {label: idx for idx, label in enumerate(self.schema)}
        self.label_map["[PAD]"] = -100  # 填充标签
        self.data = []
        self.sentences = []
        self.load()
        logging.info(f"加载完成: {len(self.data)} 个样本")

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            # for segment in segments:
            #     sentenece = []
            #     labels = []
            for segment_idx, segment in enumerate(segments):
                if not segment.strip():
                    continue

                chars = []
                orig_labels = []
                # for line in segment.split("\n"):
                #     if line.strip() == "":
                #         continue
                #     char, label = line.split()
                #     sentenece.append(char)
                #     labels.append(self.schema[label])
                # 读取字符和标签
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    char = parts[0]
                    label = parts[1]
                    chars.append(char)
                    orig_labels.append(label)
                # self.sentences.append("".join(sentenece))
                # input_ids = self.encode_sentence(sentenece)
                # labels = self.padding(labels, -1)
                # 处理整个序列
                text = "".join(chars)
                self.sentences.append(text)
                # self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
                # 编码文本 - 禁用特殊 token
                encoding = self.tokenizer.encode_plus(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_offsets_mapping=True,  # 关键：获取字符到 token 的映射
                    add_special_tokens=False  # 关键：禁用特殊 token
                )

                # 对齐标签
                aligned_labels = self.align_labels(orig_labels, encoding["offset_mapping"])

                self.data.append((
                    torch.LongTensor(encoding["input_ids"]),
                    torch.LongTensor(encoding["attention_mask"]),
                    torch.LongTensor(aligned_labels)
                ))

                # 调试输出
                if segment_idx < 3:  # 打印前3个样本
                    print(f"\n样本 {segment_idx}:")
                    print(f"原始文本: {text}")
                    print(f"原始标签: {orig_labels}")
                    print(f"Token IDs: {encoding['input_ids']}")
                    print(f"对齐标签: {aligned_labels}")
                    print(f"Offset Mapping: {encoding['offset_mapping']}")

        return

    def align_labels(self, orig_labels, offset_mapping):
        """
        将原始字符级标签对齐到 BERT tokens
        """
        aligned_labels = []
        char_idx = 0

        for offset in offset_mapping:
            start, end = offset

            # 处理填充位置 (0,0)
            if start == end == 0:
                aligned_labels.append(self.label_map["[PAD]"])
                continue

            # 处理单个字符对应单个 token 的情况
            if end - start == 1:
                if char_idx < len(orig_labels):
                    aligned_labels.append(self.label_map[orig_labels[char_idx]])
                    char_idx += 1
                else:
                    aligned_labels.append(self.label_map["O"])  # 默认 O 标签
            # 处理子词拆分（如数字序列）
            else:
                # 主 token 使用原标签
                if char_idx < len(orig_labels):
                    aligned_labels.append(self.label_map[orig_labels[char_idx]])
                    char_idx += 1
                else:
                    aligned_labels.append(self.label_map["O"])

                # 跳过当前字符的子词
                # while char_idx < len(orig_labels) and offset_mapping[char_idx] < end:
                #     # 使用 "I" 标签或忽略
                #     if "I-" in orig_labels[char_idx]:
                #         aligned_labels.append(self.label_map[orig_labels[char_idx]])
                #     else:
                #         aligned_labels.append(self.label_map["I-" + orig_labels[char_idx][2:]])
                #     char_idx += 1

        return aligned_labels

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            # 确保包含必要的标签
            if "O" not in schema:
                schema["O"] = len(schema)
        return schema

#加载字表或词表
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



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("./ner_data/train.txt", Config)
    tokenizer = BertTokenizerFast.from_pretrained(r"D:\PycharmProjects\AI学习预习\week6+语言模型和预训练\bert-base-chinese")
    # print(dg[0])
    print(dg[0])
    for batch in dg:
        input_ids, attention_mask, labels = batch
        print("\n批次形状:")
        print("输入ID:", input_ids.shape)
        print("注意力掩码:", attention_mask.shape)
        print("标签:", labels.shape)
        break

    # print(dg[1])
    # tokenizer = BertTokenizer.from_pretrained(Config["bert_path"],return_dict=True)
    # text = "武汉市长江大桥长1800米"
    # chars = list(text)
    # print(chars)
    # text_with_space = " ".join(chars)  # "武 汉 市 长 江 大 桥 长 1 8 0 0 米"
    # print(text_with_space)
    # tokens = tokenizer.tokenize(text_with_space)
    # print(tokens)