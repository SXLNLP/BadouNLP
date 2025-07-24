# -*- coding: utf-8 -*-

import json
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

"""
数据加载，CSV文件处理
"""

def process_csv_to_json(csv_file_path="文本分类练习.csv", test_size=0.2):
    """
    处理CSV文件，转换为JSON格式并划分训练集和验证集
    """
    print(f"正在处理CSV文件: {csv_file_path}")

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='gbk')

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 第一列是标签，第二列是文本
    df.columns = ['label', 'review']

    # 清理数据
    df = df.dropna()
    df = df.drop_duplicates()

    # 统计信息
    print(f"数据量: {len(df)}")
    print(f"标签分布:")
    print(df['label'].value_counts())

    # 划分训练集和验证集
    train_df, valid_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])

    # 创建data目录
    if not os.path.exists('data'):
        os.makedirs('data')

    # 保存为JSON格式
    def save_to_json(data, filename):
        with open(filename, 'w', encoding='utf8') as f:
            for _, row in data.iterrows():
                tag = "好评" if row['label'] == 1 else "差评"
                item = {
                    "tag": tag,
                    "title": str(row['review'])
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_to_json(train_df, 'data/train_tag_news.json')
    save_to_json(valid_df, 'data/valid_tag_news.json')

    print(f"训练集保存到: data/train_tag_news.json ({len(train_df)} 条)")
    print(f"验证集保存到: data/valid_tag_news.json ({len(valid_df)} 条)")

    return len(train_df), len(valid_df)


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = {'差评': 0, '好评': 1}
        self.config["class_num"] = len(self.index_to_label)

        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        else:
            self.vocab = load_vocab(config["vocab_path"])
            self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     padding='max_length', truncation=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
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
            token_dict[token] = index + 1   #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    # 处理CSV文件
    csv_file = "文本分类练习.csv"
    if os.path.exists(csv_file):
        process_csv_to_json(csv_file)
    else:
        print(f"找不到文件: {csv_file}")