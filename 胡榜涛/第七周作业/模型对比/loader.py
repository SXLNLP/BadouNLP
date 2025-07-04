# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = [0,1]
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])#实例化一个对象
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()



    def load(self):
        self.data = []
        df = pd.read_csv(self.path, encoding='utf-8')
        for row in df.itertuples():
            label=row.label
            review=row.review
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                #为什么这里不直接在词表里找呢，应为bert有个cls和sep，用这个函数可以自动返回
            else:
                input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])#这里就是代表了输入向量和目标分类
        return

    def encode_sentence(self, text):
        input_id = []
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
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


def csv_data_load(config):
    data_path=config["orign_data_path"]
    dg=pd.read_csv(data_path,encoding='utf-8')
    sample_size=config['divide_test_sample_size']
    # 筛选正样本和负样本
    positive_samples = dg[dg['label']==1]
    negative_samples = dg[dg['label']==0]

    # 各抽取指定数量的样本
    selected_positive = positive_samples.sample(min(sample_size, len(positive_samples)))
    selected_negative = negative_samples.sample(min(sample_size, len(negative_samples)))

    # 合并筛选出的正负样本
    selected_combined = pd.concat([selected_positive, selected_negative])

    # 保存合并后的样本
    selected_combined.to_csv(config["valid_data_path"], index=False)

    # 获取剩余的样本索引
    selected_indices = pd.Index(selected_combined.index)

    # 计算剩余样本（未被选中的样本）
    remaining_samples = dg[~dg.index.isin(selected_indices)]

    # 保存剩余样本
    remaining_samples.to_csv(config["train_data_path"], index=False)



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
