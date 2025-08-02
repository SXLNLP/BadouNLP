# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/7/21
# @Author      : liuboyuan
import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from collections import defaultdict
import random
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, data_type="test"):
        self.config = config
        self.path = data_path
        # News classification label mapping
        LABELS = ['文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '军事', '旅游', '国际', '证券', '农业', '电竞', '民生']
        self.index_to_label = {idx: name for idx, name in enumerate(LABELS)}
        self.label_to_index = {name: idx for idx, name in self.index_to_label.items()}
        self.schema = self.label_to_index
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load(data_type)


    def load(self, data_type="test"):
        """
        Load dataset with each line being a JSON object:
        {"text": "...", "label_name": "...", "label": int}
        """
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                title = sample["text"]
                # support both integer label id and label name
                if "label" in sample:
                    label = int(sample["label"])
                else:
                    label_name = sample.get("label_name")
                    label = self.label_to_index.get(label_name, -1)
                if label not in self.index_to_label:
                    # skip unknown label
                    continue
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(
                        title,
                        max_length=self.config["max_length"],
                        truncation=True,
                        padding='max_length'
                    )
                else:
                    input_id = self.encode_sentence(title)
                self.sentences.append(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                if data_type == 'train':
                    self.data_type = 'train'
                    self.knwb[label].append(input_id)
                self.data.append([input_id, label_index])
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
    
    #随机生成3元组样本，2正1负
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 先选定两个意图，之后从第一个意图中取2个问题，第二个意图中取一个问题
        p, n = random.sample(standard_question_index, 2)
        # 如果某个意图下刚好只有一条问题，那只能两个正样本用一样的；
        # 这种对训练没帮助，因为相同的样本距离肯定是0，但是数据充分的情况下这种情况很少
        if len(self.knwb[p]) == 1:
            s1 = s2 = self.knwb[p][0]
        #这应当是一般情况
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
        # 随机一个负样本
        s3 = random.choice(self.knwb[n])
        # 前2个相似，后1个不相似，不需要额外在输入一个0或1的label，这与一般的loss计算不同
        return [s1, s2, s3]

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True, data_type="test"):
    dg = DataGenerator(data_path, config, data_type)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])