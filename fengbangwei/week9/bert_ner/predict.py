# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizerFast

"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.tokenizer = BertTokenizerFast.from_pretrained(config["pretrain_model_path"])

        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        text = list(sentence)
        encoding = self.tokenizer(
            text,
            max_length=self.config["max_length"],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
            is_split_into_words=True
        )
        input_ids = encoding['input_ids'].flatten().unsqueeze(0) #添加1维度

        with torch.no_grad():
            res = self.model(input_ids)[0]
            res = torch.argmax(res, dim=-1)
        labeled_sentence = ""
        # for char, label_index in zip(sentence, res):
        #     labeled_sentence += char + self.index_to_sign[int(label_index)]

        # 获取 offset_mapping 来判断哪些是真实字符，哪些是 [CLS]/[SEP]
        offset_mapping = encoding.offset_mapping[0].tolist()
        # "B-PERSON": 2,
        # "I-PERSON": 6,
        # tensor([8, 8, 2, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        #         8, 8, 8, 8])

        for idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # 忽略 [PAD]
                continue
            if start == end:  # 忽略 [CLS], [SEP] 这类无实际字符映射的 token
                continue
            try:
                label_index = int(res[idx])
                label = self.index_to_sign[label_index]
                labeled_sentence += sentence[idx-1] + label
            except IndexError:
                continue
        return labeled_sentence

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_30.pth")

    sentence = "联合国秘书长地球越来越危险热"
    res = sl.predict(sentence)
    print(res)

    sentence = "当日10小时60余次地震日本紧急开记者会"
    res = sl.predict(sentence)
    print(res)

    sentence = "日本末世预言剩1天有人凌晨紧急撤离"
    res = sl.predict(sentence)
    print(res)

    sentence = "中国回应美将向不同国家征收新关税"
    res = sl.predict(sentence)
    print(res)
