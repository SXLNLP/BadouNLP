import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
数据加载（支持三元组样本生成）
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])  # {意图名: id}
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None  # "train" or "test"
        self.knwb = defaultdict(list)  # {意图id: [输入id列表]}
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集（构建三元组）
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    label_id = self.schema[label]  # 转换为意图id
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[label_id].append(input_id)
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.generate_triplet_sample()  # 生成三元组样本
        else:
            return self.data[index]

    def generate_triplet_sample(self):
        """生成三元组样本 (anchor, positive, negative)"""
        intent_ids = list(self.knwb.keys())
        if not intent_ids:
            raise ValueError("训练数据为空")

        # 随机选择锚点意图
        anchor_intent = random.choice(intent_ids)
        # 确保锚点意图有足够样本
        if len(self.knwb[anchor_intent]) < 2:
            return self.generate_triplet_sample()

        # 生成锚点和正例（同意图）
        anchor, positive = random.sample(self.knwb[anchor_intent], 2)

        # 生成负例（不同意图）
        while True:
            neg_intent = random.choice(intent_ids)
            if neg_intent != anchor_intent and len(self.knwb[neg_intent]) > 0:
                negative = random.choice(self.knwb[neg_intent])
                break

        return [anchor, positive, negative]  # 三元组样本


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding
    # 补充特殊符号
    if "[UNK]" not in token_dict:
        token_dict["[UNK]"] = len(token_dict) + 1
    return token_dict


def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        schema = json.loads(f.read())
    return schema  # {意图名: id}


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    return DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)