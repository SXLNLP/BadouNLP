# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer
from transformers import BertTokenizerFast

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config["pretrain_model_path"])
        # self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"],add_special_tokens=True)

        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load1(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            # 段落换行
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                # 每个字是 一行一行的
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    # 按空格切分
                    char, label = line.split()
                    sentence.append(char)  # 中 B-ORGANIZATION
                    labels.append(self.schema[label])  # B-ORGANIZATION -> 1
                self.sentences.append("".join(sentence))

                encoded = self.tokenizer.encode_plus(
                    "".join(sentence),
                    max_length=self.config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]  # 必须传入模型
                word_ids = encoded.word_ids()
                aligned_labels = []
                for w_id in word_ids:
                    if w_id is None or w_id >= len(labels):
                        aligned_labels.append(-1)
                    else:
                        aligned_labels.append(labels[w_id])
                # aligned_labels = self.padding(labels, -1)

                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(attention_mask),
                                  torch.LongTensor(aligned_labels)
                                  ])
        return

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            # 段落换行
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                # 每个字是 一行一行的
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    # 按空格切分
                    char, label = line.split()
                    sentence.append(char)  # 中 B-ORGANIZATION
                    labels.append(self.schema[label])  # B-ORGANIZATION -> 1
                self.sentences.append("".join(sentence))
                # 对齐标签
                encoding = self.tokenizer(
                    sentence,
                    max_length=self.config["max_length"],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_offsets_mapping=True,
                    is_split_into_words=True
                )

                # 创建list存储标签 -100 忽略
                label_ids = []
                word_ids = encoding.word_ids()
                previous_word_idx = None

                for word_idx in word_ids:
                    # Special tokens get label -100 (ignored by loss function)
                    if word_idx is None:
                        label_ids.append(-1)
                    # Only label the first token of a given word
                    elif word_idx != previous_word_idx:
                        label_ids.append(labels[word_idx])
                    else:
                        label_ids.append(-1)
                    previous_word_idx = word_idx
                input_ids = encoding['input_ids'].flatten()  # 16 1 100 变成 16 100
                attention_mask = encoding['attention_mask'].flatten()  # 16 1 100 变成 16 100

                self.data.append(
                    [torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(label_ids)])

        return


    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            # 因为输入序列的词汇表索引通常从1开始（0预留给填充）。
            input_id = self.padding(input_id)
        return input_id


    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = [pad_token] + input_id[:self.config["max_length"] - 2] + [pad_token]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]


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
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("ner_data/train", Config)
