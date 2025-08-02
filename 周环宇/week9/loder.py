# -*- coding: utf-8 -*-

import json  # 导入json模块，用于处理JSON格式数据
import re  # 导入正则表达式模块，用于文本模式匹配
import os  # 导入操作系统接口模块，用于文件和目录操作
import torch  # 导入PyTorch深度学习框架
import random  # 导入随机数模块，用于数据打乱等操作
import jieba  # 导入中文分词模块，用于文本分词
import numpy as np  # 导入numpy库，用于数值计算
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器类
from transformers import BertTokenizer

"""
数据加载
"""

class DataGenerator:
    def __init__(self, data_path, config):  # 构造函数
        self.config = config  # 保存配置
        self.path = data_path  # 保存数据路径
        self.vocab = load_vocab(config["vocab_path"])  # 加载词汇表
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.config["vocab_size"] = len(self.vocab)  # 将词汇表大小存入配置
        self.sentences = []  # 初始化句子列表
        self.schema = self.load_schema(config["schema_path"])  # 加载标签模式
        self.o_label = self.schema.get("O", 8)
        self.load()  # 加载数据

    def load(self):  # 加载数据方法
        self.data = []  # 初始化数据列表
        with open(self.path, encoding="utf8") as f:  # 打开数据文件
            segments = f.read().split("\n\n")  # 按段落分割数据
            for segment in segments:  # 处理每个段落
                sentenece = []  # 初始化句子列表
                labels = []  # 初始化标签列表
                for line in segment.split("\n"):  # 按行分割段落
                    if line.strip() == "":  # 跳过空行
                        continue
                    char, label = line.split()  # 分割字符和标签
                    sentenece.append(char)  # 添加字符
                    labels.append(self.schema[label])  # 添加对应的标签ID
                self.sentences.append("".join(sentenece))  # 合并成完整句子
                encoding = self.tokenizer.encode_plus(
                    "".join(sentenece),
                    add_special_tokens=True,  # 添加[CLS]和[SEP]
                    max_length=self.config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"  # 返回PyTorch张量
                )
                input_ids = encoding["input_ids"].squeeze(0)  # 去除batch维度
                labels = [self.o_label] + labels + [self.o_label]  # 填充或截断标签
                # 确保 labels 长度与 input_ids 一致
                if len(labels) < input_ids.shape[0]:
                    labels += [self.o_label] * (input_ids.shape[0] - len(labels))
                else:
                    labels = labels[:input_ids.shape[0]]
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])  # 添加到数据列表
        return

    def encode_sentence(self, text, padding=True):  # 编码句子为ID序列
        input_id = []  # 初始化ID列表
        if self.config["vocab_path"] == "words.txt":  # 如果使用词级别
            for word in jieba.cut(text):  # 对文本进行分词
                # 获取词的ID，如果不存在则使用[UNK]的ID
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:  # 如果使用字符级别
            for char in text:  # 处理每个字符
                # 获取字符的ID，如果不存在则使用[UNK]的ID
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:  # 是否进行填充
            input_id = self.padding(input_id)  # 填充或截断
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):  # 填充或截断序列
        # 截断到最大长度
        input_id = input_id[:self.config["max_length"]]
        # 填充到最大长度
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):  # 返回数据集大小
        return len(self.data)

    def __getitem__(self, index):  # 获取指定索引的数据项
        return self.data[index]

    def load_schema(self, path):  # 加载标签模式
        with open(path, encoding="utf8") as f:  # 打开模式文件
            return json.load(f)  # 加载JSON数据

# 加载字表或词表
def load_vocab(vocab_path):  # 加载词汇表函数
    token_dict = {}  # 初始化词汇表字典
    with open(vocab_path, encoding="utf8") as f:  # 打开词汇表文件
        for index, line in enumerate(f):  # 逐行读取
            token = line.strip()  # 去除两端空白字符
            # 为每个token分配ID，0保留给padding
            token_dict[token] = index + 1
    return token_dict

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):  # 加载数据函数
    dg = DataGenerator(data_path, config)  # 创建数据生成器
    # 使用PyTorch的DataLoader封装数据
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl  # 返回数据加载器


if __name__ == "__main__":  # 程序入口
    from config import Config  # 导入配置
    dg = DataGenerator("../ner_data/train.txt", Config)  # 创建数据生成器实例
