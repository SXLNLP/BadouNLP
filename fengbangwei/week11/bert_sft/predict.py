# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from transformers import BertTokenizer

from deepseek.week11.lstm_opt.main import LanguageModel, generate_sentence

"""
模型效果测试
"""


class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.model = LanguageModel(len(self.tokenizer.vocab))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        print("模型加载完毕!")

    def predict(self, sentence):
        labeled_sentence = generate_sentence(sentence, self.model, self.tokenizer)
        return labeled_sentence


if __name__ == "__main__":
    sl = SentenceLabel(Config, r"output/epoch_99.pth")

    sentence = "地球公转周期是多久"
    res = sl.predict(sentence)
    print(res)

    sentence = "水由什么元素组成"
    res = sl.predict(sentence)
    print(res)

    sentence = "光速是多少"
    res = sl.predict(sentence)
    print(res)

    sentence = "AI是什么意思"
    res = sl.predict(sentence)
    print(res)

    sentence = "空气中氧气含量多少"
    res = sl.predict(sentence)
    print(res)