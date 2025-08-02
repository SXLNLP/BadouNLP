# -*- coding: utf-8 -*-

import json
import random
from collections import defaultdict

import jieba
import torch
from torch.utils.data import DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        self.train_data_size = config['epoch_data_size']  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  # 用来标识加载的是训练集还是测试集 'train' or 'test'
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = 'train'
                    questions = line['questions']
                    label = line['target']
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    self.data_type = 'test'
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config['vocab_path'] == 'words.txt':
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab['[UNK]']))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == 'train':
            return self.config['epoch_data_size']
        else:
            assert self.data_type == 'test', self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == 'train':
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())

        # 随机选择一个标准问题作为Anchor和Positive的源
        p = random.choice(standard_question_index)

        # 如果当前标准问题下样本不足2个，重新选择
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()

        # 从其他标准问题中选择Negative的源
        others = [idx for idx in standard_question_index if idx != p]
        n = random.choice(others)

        # 从Positive源中随机选一个样本作为Anchor
        anchor = random.choice(self.knwb[p])

        # 从Positive源中随机选另一个样本作为Positive
        positive = random.choice([x for x in self.knwb[p] if not torch.equal(x, anchor)])

        # 从Negative源中随机选一个样本作为Negative
        negative = random.choice(self.knwb[n])

        # 返回三元组 (Anchor, Positive, Negative)
        return [anchor, positive, negative]


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding='utf8') as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    dg = DataGenerator('./data/valid.json', Config)
    print(dg[1])
