
import csv
import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

"""
数据加载与划分
"""


class DataGenerator(Dataset):
    def __init__(self, data_path, config, is_train=True, indices=None):
        self.config = config
        self.path = data_path
        self.is_train = is_train
        self.indices = indices  # 用于划分训练集和测试集的索引

        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.data = []
        self.labels = []  # 保存所有标签，用于划分
        self.load()

        # 如果指定了索引，只使用对应的数据
        if indices is not None:
            self.data = [self.data[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def load(self):
        with open(self.path, 'r', encoding="utf8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                y = row['label']
                x = row['review']

                # 保存原始标签（用于划分）
                self.labels.append(y)

                if self.config["model_type"] == "bert":
                    x = self.tokenizer.encode(x, max_length=self.config["max_length"],
                                              padding='max_length',  # 填充到固定长度
                                              truncation=True,return_attention_mask=True)
                else:
                    x = self.encode_sentence(x)
                x = torch.LongTensor(x)
                y = torch.LongTensor([int(y)])  # 转换为整数张量
                self.data.append([x, y])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

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
            token_dict[token] = index + 1  # 0留给padding位置
    return token_dict


def split_dataset(dataset, test_size=0.2, stratify=None, random_state=42):
    """
    划分数据集为训练集和测试集
    - dataset: 原始数据集
    - test_size: 测试集比例
    - stratify: 分层划分的标签
    - random_state: 随机种子，保证可复现
    """
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    return train_indices, test_indices


def load_data(data_path, config, test_size=0.2, shuffle=True):
    """加载数据并划分为训练集和测试集"""
    # 1. 加载完整数据集
    full_dataset = DataGenerator(data_path, config)

    # 2. 划分数据集（推荐分层划分）
    train_indices, test_indices = split_dataset(
        full_dataset,
        test_size=test_size,
        stratify=full_dataset.labels,  # 按标签分层
        random_state=config.get("random_seed", 42)
    )

    # 3. 创建训练集和测试集
    train_dataset = DataGenerator(data_path, config, indices=train_indices)
    test_dataset = DataGenerator(data_path, config, is_train=False, indices=test_indices)

    # 4. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    from config import Config

    file_path = r'F:\1\八斗精品班\第七周 文本分类\week7 文本分类问题\week7 文本分类问题\文本分类练习.csv'
    #config = Config()  # 假设Config类已定义batch_size等参数

    # 加载并划分数据
    train_loader, test_loader = load_data(file_path, Config)

    # 验证划分结果
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 查看训练集第一个样本
    print("训练集第一个样本:")
    print(train_loader.dataset[0])

  
