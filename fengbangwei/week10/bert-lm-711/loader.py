# -*- coding: utf-8 -*-

import torch
import random
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, window_size):
        self.config = config
        self.path = data_path
        self.window_size = window_size
        self.vocab = build_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.corpus = load_corpus(data_path)
        self.load()

    # 随机生成一个样本
    # 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
    def load(self):
        self.data = []
        start = random.randint(0, len(self.corpus) - 1 - self.window_size)
        end = start + self.window_size
        window = self.corpus[start:end]
        target = self.corpus[start + 1:end + 1]  # 输入输出错开一位
        # print(window, target)
        x = [self.vocab.get(word, self.vocab["<UNK>"]) for word in window]  # 将字转换成序号
        y = [self.vocab.get(word, self.vocab["<UNK>"]) for word in target]
        self.data.append([torch.LongTensor(x), torch.LongTensor(y)])
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, window_size, shuffle=True):
    dg = DataGenerator(data_path, config, window_size)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("corpus.txt", Config, 10)