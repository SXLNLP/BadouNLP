# -*- coding: utf-8 -*-

import json
import re
import os

import pandas as pd
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
        """ 原先是多分类，这次要改成二分类。0代表差评  1代表好评 """
        self.index_to_label = {0: 0, 1: 1}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        # 将纯粹的词表，转换成词表token文件，每个词有一个唯一的token对应
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        #给data赋值，将每条样本数据转换成两个向量，一个向量是文本对应的token序号，一个是标签对应的index
        """
        [
            [tensor([ 101, 1184, 5683, 3215, 2571, 7623, 2421, 2802, 2339, 3198, 5959, 8132, 1039, 3295,  680, 3448, 3308, 
        836, 1394,  868,  102,    0,    0,    0,  0,    0,    0,    0,    0,    0]),    tensor([14])],
            [tensor([ 101, 1198, 358, 3215, 2571, 7623, 2421, 2802, 2339, 3198, 5959, 8132, 1039, 3295,  680, 3448, 3308, 
        836, 1394,  868,  102,    0,    0,    0,  0,    0,    0,    0,    0,    0]),    tensor([18])]
        ]
        """

        # self.load_json()
        self.load_csv()


    """ 读取csv格式的样本数据集 """
    def load_csv(self):
        self.data = []
        chunk_size = 1000 #按块读取文件，适合于大数据集文件
        for chunk in pd.read_csv(self.path, chunksize=chunk_size, encoding="utf-8"):
            for index, row in chunk.iterrows():
                tag = row["label"]
                title = row["review"]

                label = self.label_to_index[tag]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    """ 读取json格式的样本数据集 """
    def load_json(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
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

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

#将纯粹的词表，转换成词表token文件，每个词有一个唯一的token对应
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
    dg = DataGenerator("../data/valid_tag_news.json", Config)
    print(dg[1])
