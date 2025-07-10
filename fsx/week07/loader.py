import csv
import json
import random

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from nn_pipline2 import config
from nn_pipline2.config import Config


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.train_path = config["train_data_path"]
        self.index_to_label = {0: "差评", 1: "好评"}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.train_data = None
        self.test_data = None
        self.data = None

        if self.config["is_just_train_data"]:
            self.train_data, self.test_data = get_train_test_data_by_train_csv(self.config)
        if self.train_data is not None and self.test_data is not None:
            self.train_data = self.load_dict_data(self.train_data)
            self.test_data = self.load_dict_data(self.test_data)
            self.data = self.train_data
        else:
            if self.config["is_csv"]:
                self.data = self.load_dict_data(csv_to_dict(self.config["train_data_path"]))
            else:
                self.data = self.load_json_data()

    def load_json_data(self):
        self.data = []
        with open(self.train_path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return self.data

    def load_dict_data(self, dict_data):
        self.data = []

        for input, label in dict_data.items():
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(input, max_length=self.config["max_length"],
                                                 pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(input)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([int(label)])
            self.data.append([input_id, label_index])

        return self.data

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


def get_train_test_data_by_train_csv(config):
    file_path = config["train_data_path"]
    train_ratio = config.get("train_ratio", 0.5)  # 默认50%训练数据
    seed = config.get("seed", 42)  # 默认随机种子

    # 验证训练比例是否有效
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio必须在(0.0, 1.0)范围内")

    # 从CSV文件读取所有数据
    all_data = csv_to_dict(file_path)

    # 获取所有唯一的评论（字典的键）
    reviews = list(all_data.keys())

    # 设置随机种子，确保结果可复现
    random.seed(seed)

    # 随机打乱评论列表
    random.shuffle(reviews)

    # 计算分割点
    split_index = int(len(reviews) * train_ratio)

    # 划分训练集和测试集的评论
    train_reviews = reviews[:split_index]
    test_reviews = reviews[split_index:]

    # 从原始数据中提取对应的训练集和测试集
    train_data = {review: all_data[review] for review in train_reviews}
    test_data = {review: all_data[review] for review in test_reviews}

    return train_data, test_data


def csv_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                label = row[0].strip()  # 标签（假设在第一列）
                review = row[1].strip()  # 评论内容（假设在第二列）
                data_dict[review] = label  # 以评论为键，标签为值
    return data_dict


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def generate_data(config):
    return DataGenerator(config)


def load_data(config, shuffle=True):
    dg = DataGenerator(config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    dg = DataGenerator(Config)
    dl = DataLoader(dg, batch_size=128, shuffle=True)
    print(dg[1])
