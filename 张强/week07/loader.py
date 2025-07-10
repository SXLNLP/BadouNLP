# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
        #                        5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
        #                        10: '体育', 11: '科技', 12: '汽车', 13: '健康',
        #                        14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        # self.config["class_num"] = len(self.index_to_label)
        self.config["class_num"] = 2
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 使用正确的字段名：label 和 review
                label = line["label"]
                review = line["review"]  # 注意：这里是review字段
                if label not in [0, 1]:
                    print(f"⚠️ 无效标签 {label}，跳过此条数据")
                    continue

                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],padding="max_length",truncation=True)
                    # encoding = self.tokenizer.encode_plus(
                    #     review,
                    #     max_length=self.config["max_length"],
                    #     padding="max_length",
                    #     truncation=True,
                    #     return_attention_mask=True,
                    #     return_tensors="pt"
                    # )
                    # input_id = encoding["input_ids"].squeeze(0)
                    # attention_mask = encoding["attention_mask"].squeeze(0)
                else:
                    input_id = self.encode_sentence(review)
                    # input_id = torch.LongTensor(input_id)
                    # attention_mask = (input_id != 0).long()
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
                # self.data.append([input_id, attention_mask,label_index])
        print(f"✅ 成功加载 {len(self.data)} 条数据")
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab.get("[UNK]", 0)))
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

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("./data/文本分类练习_valid.json", Config)
    print(dg[1])
