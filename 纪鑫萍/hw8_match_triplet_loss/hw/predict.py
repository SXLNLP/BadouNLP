# -*- coding: utf-8 -*-
import jieba
import torch
from loader import load_data
from config import Config
from model import SiameseNetwork, choose_optimizer

"""
模型效果测试
"""

class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model
        self.train_data = knwb_data
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.knwb_to_vector()

    #将知识库中的问题向量化，为匹配做准备
    #每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)  # 将知识库中的问题ID转换为一个张量矩阵
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            #将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    def predict(self, sentence):
        input_id = self.encode_sentence(sentence)  # 将输入的句子转换为ID序列(通过词表)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            test_question_vector = self.model(input_id)  # 不输入labels，使用模型当前参数进行预测
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
                # ==> 计算测试问题向量与知识库中所有问答对的相似度 (计算测试问题与知识库中所有问答对的余弦相似度，并将结果存储在res中)
                    # ==> test_question_vector.unsqueeze(0)：将测试问题的向量增加一个维度，使其从形状 [dim] 变为 [1, dim]，以便进行矩阵乘法。
                    # ==> self.knwb_vectors.T：将知识库中所有问答对中的向量进行转置，使得每一列代表一个问答对的向量。
                    # ==> torch.mm(...)：执行矩阵乘法，结果是一个形状为 [1, num_knwb] 的张量，其中每个元素表示测试问题与对应问答对的相似度。
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
        return self.index_to_standard_question[hit_index]

if __name__ == "__main__":
    knwb_data = load_data(Config["train_data_path"], Config)
    model = SiameseNetwork(Config)
    model.load_state_dict(torch.load("model_output/epoch_10.pth"))
    pd = Predictor(Config, model, knwb_data)

    while True:
        # sentence = "固定宽带服务密码修改"
        sentence = input("请输入问题：")
        res = pd.predict(sentence)
        print(res)

