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
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []  # 创建空列表self.data存储原始数据
        self.knwb = defaultdict(list)  # 使用defaultdict(list)构建知识库容器self.knwb，支持按标签自动创建列表
        with open(self.path, encoding="utf8") as f:  # 以UTF-8编码打开指定路径文件
            for line in f:
                line = json.loads(line)  # 逐行读取并使用json.loads解析JSON格式数据
                # 加载训练集
                if isinstance(line, dict):  # 通过isinstance检查字典类型数据
                    self.data_type = "train"  # 标记self.data_type为训练集模式
                    questions = line["questions"]  # 提取问题列表questions和目标标签label
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)  # 调用encode_sentence方法将文本转为数值ID序列
                        input_id = torch.LongTensor(input_id)  # 转换为torch.LongTensor张量格式
                        self.knwb[self.schema[label]].append(input_id)  # 按schema映射的标准标签分类存储到知识库
                # 加载测试集
                else:
                    self.data_type = "test"  # 设置self.data_type为"test"标识测试集模式
                    assert isinstance(line, list)  # 使用assert isinstance确保输入数据为列表格式（数据格式校验）
                    question, label = line
                    input_id = self.encode_sentence(question)  # 调用encode_sentence方法将问题文本转为数值ID序列
                    input_id = torch.LongTensor(input_id)  # 通过torch.LongTensor转换为PyTorch长整型张量
                    label_index = torch.LongTensor([self.schema[label]])  # 通过schema字典将文本标签映射为数值索引
                    self.data.append(
                        [input_id, label_index])  # 将处理后的输入特征input_id和标签label_index组成二元组添加到self.data列表中形成标准数据集结构
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

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]

    def random_train_sample(self):
        """生成用于pair-wise loss训练的三元组"""
        valid_cats = [k for k,v in self.knwb.items() if len(v)>=2]
        if len(valid_cats) < 2:
            raise ValueError(f"需要至少2个包含足够样本的类别")

        # valid_cats = [k for k in self.schema if k in self.data]
        # if len(valid_cats) < 2:
        #     raise ValueError("有效类别不足2个")
        main_cat = random.choice(valid_cats)
        contrast_cat = random.choice([k for k in valid_cats if k != main_cat]) #随机选择主类别和对比类别
        # 从主类别选两条相似文本
        anchor, positive = random.sample(self.knwb[main_cat], 2)
        negative = random.choice(self.knwb[contrast_cat])
        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
        }


#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid.json", Config)
    print(dg[1])
