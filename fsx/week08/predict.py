# -*- coding: utf-8 -*-
import jieba
import torch
import numpy as np
from loader import load_data
from config import Config
from model import SentenceEncoder
from triplet import config

"""
模型效果测试（含评估功能）
"""


class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model
        self.train_data = knwb_data
        if config["gpu_switch"]:
            self.model = model.to("mps")
        self.model.eval()
        self.knwb_to_vector(config)
        # 新增：构建知识库句子到索引的映射（用于评估）
        self.build_sentence_to_index_map()

    # 将知识库中的问题向量化
    def knwb_to_vector(self, config):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if config["gpu_switch"]:
                self.model = model.to("mps")
            self.knwb_vectors = self.model(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    # 新增：构建知识库句子到索引的映射（用于评估时查找正确句子位置）
    def build_sentence_to_index_map(self):
        self.sentence_to_knwb_index = {}
        # 遍历知识库所有句子，记录其在question_ids中的索引
        for idx, question_id in enumerate(self.question_ids):
            # 将tensor转为tuple作为key（tensor不能作为字典key）
            question_tuple = tuple(question_id.cpu().numpy())
            self.sentence_to_knwb_index[question_tuple] = idx

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    def predict(self, sentence, config):
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if config["gpu_switch"]:
            self.model = model.to("mps")
        with torch.no_grad():
            test_question_vector = self.model(input_id)
            res = torch.mm(test_question_vector, self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))
            hit_index = self.question_index_to_standard_question_index[hit_index]
        return self.index_to_standard_question[hit_index]


if __name__ == "__main__":
    knwb_data = load_data(Config["train_data_path"], Config)
    model = SentenceEncoder(Config)
    device = torch.device('cpu')
    model.load_state_dict(torch.load("model_output/epoch_50.pth", map_location=device))
    pd = Predictor(Config, model, knwb_data)

    while True:
        # sentence = "固定宽带服务密码修改"
        sentence = input("请输入问题：")
        res = pd.predict(sentence, Config)
        print(res)
