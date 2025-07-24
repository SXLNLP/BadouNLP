# -*- coding: utf-8 -*-

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

# 另外的主要修改部分为model.py中的TorchModel部分
# class TorchModel(nn.Module):
#     def __init__(self, config):
#         super(TorchModel, self).__init__()
#         class_num = config["class_num"]
#         num_layers = config["num_layers"]
#         bilstm_hidden_size = config["bilstm_hidden_size"]
#         dropout = config["bilstm_dropout"]
#         # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
#         # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
# 
#         self.bert = BertModel.from_pretrained(config["bert_path"])
# 
#         self.bilstm = nn.LSTM(
#             input_size=self.bert.config.hidden_size,
#             hidden_size=bilstm_hidden_size,
#             num_layers=num_layers,
#             bidirectional=True,
#             batch_first=True,
#             dropout=dropout)
#         self.classify = nn.Linear(bilstm_hidden_size * 2, class_num)
#         self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
#         self.crf_layer = CRF(class_num, batch_first=True)
#         self.use_crf = config["use_crf"]
#         self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失
# 
#     # 当输入真实标签，返回loss值；无真实标签，返回预测值
#     def forward(self, x, attention_mask=None, target=None):
#         # x = self.embedding(x)  # input shape:(batch_size, sen_len)
#         # x, _ = self.layer(x)      # input shape:(batch_size, sen_len, input_dim)
#         outputs = self.bert(
#             x,
#             attention_mask=attention_mask if attention_mask is not None else (x != 0).long())
# 
#         sequence_output = outputs[0]
#         predict = self.classify(sequence_output) 
# 
#         if target is not None:
#             if self.use_crf:
#                 mask = target.gt(-1) 
#                 return - self.crf_layer(predict, target, mask, reduction="mean")
#             else:
#                 # (number, class_num), (number)
#                 return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
#         else:
#             if self.use_crf:
#                 return self.crf_layer.decode(predict)
#             else:
#                 return predict
if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

