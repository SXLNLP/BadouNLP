# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, sample_count):
        self.config = config
        self.path = data_path
        self.sample_count = sample_count
        self.index_to_label = {'差评': 0, '好评': 1}
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        # 统计样本个数
        count = 0
        with open(self.path, encoding="utf8") as f:
            for line in f:
                # 样本数限制
                if self.sample_count is not None and count >= self.sample_count:
                    break
                line = json.loads(line)
                # tag = line["tag"] # 0 1
                label = line["tag"]  # 0 1
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
                count += 1
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
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
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True, sample_count=None):
    dg = DataGenerator(data_path, config, sample_count)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            # json.dump(train_df, f, ensure_ascii=False, indent=2) 美化效果
            f.write('\n')  # 换行


if __name__ == "__main__":
    from config import Config

    # dg = DataGenerator("valid_tag_news.json", Config)
    # print(dg[1])

    # 数据分析
    df = pd.read_csv('D:/AI/ai_project/deepseek/week7/data/文本分类练习.csv', encoding='utf-8')
    # # print(df)  # 输出DataFrame结构化数据
    # total = df.count()
    # print(total)
    # correct = df["label"].sum()
    # avg_len = df['review'].str.len().mean()
    # print("总数", total)
    # print("正样本数", correct)
    # print("负样本数", total - correct)
    # print("文本平均长度", avg_len)

    # 训练集/验证集划分
    # {"tag": "文化", "title": "“少年不可欺”，版权亦不可欺"}
    data = []
    for value in df.values:
        data.append({"tag": value[0], "title": value[1]})
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    # print(train_df)
    # print(test_df)

    # 写入训练集 JSON 文件
    write_jsonl('D:/AI/ai_project/deepseek/week7/data/train_tag_review.json', train_df)

    write_jsonl('D:/AI/ai_project/deepseek/week7/data/valid_tag_review.json', test_df)
