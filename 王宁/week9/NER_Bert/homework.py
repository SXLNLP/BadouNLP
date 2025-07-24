#周末考试 待补充完成
import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


# 自定义一个Tokenizer处理数字字符，将多位数字分割
class CustomBertTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tokenize(self, text):
        tokens = []
        for char in text:
            if char.isdigit():
                # 分割数字字符
                tokens.extend(super()._tokenize(f"{char}"))
            else:
                tokens.extend(super()._tokenize(char))
        return [token.strip() for token in tokens if token.strip()]

    def tokenize(self, text, **kwargs):
        tokens = self._tokenize(text)
        return [token if token not in self.all_special_tokens else token for token in tokens]


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = CustomBertTokenizer.from_pretrained(config["bert_path"], do_lower_case=False)
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                encoded = self.tokenizer(
                    "".join(sentence),
                    padding="max_length",
                    max_length=self.config["max_length"],
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
                self.sentences.append("".join(sentence))
                labels = self.padding(labels, -1)
                self.data.append([input_ids, attention_mask, torch.LongTensor(labels)])
        return

    def encode_sentence(self, text):
        encoded = self.tokenizer("".join(text), padding="max_length",
                                 max_length=self.config["max_length"],
                                 truncation=True,
                                 return_offsets=True,
                                 return_tensors="pt")
        return encoded

    # def encode_sentence(self, text, padding=True):
    #     input_id = []
    #     if self.config["vocab_path"] == "words.txt":
    #         for word in jieba.cut(text):
    #             input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
    #     else:
    #         for char in text:
    #             input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
    #     if padding:
    #         input_id = self.padding(input_id)
    #     return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)
