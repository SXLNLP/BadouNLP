# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "/root/autodl-tmp/week08/data/schema.json",
    "train_data_path": "/root/autodl-tmp/week08/data/train.json",
    "valid_data_path": "/root/autodl-tmp/week08/data/valid.json",
    "vocab_path":"/root/autodl-tmp/week08/data/chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                #加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
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
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]

    #依照一定概率生成负样本或正样本
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        # standard_question_index = list(self.knwb.keys())
        # #随机正样本
        # if random.random() <= self.config["positive_sample_rate"]:
        #     p = random.choice(standard_question_index)
        #     #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        #     if len(self.knwb[p]) < 2:
        #         return self.random_train_sample()
        #     else:
        #         s1, s2 = random.sample(self.knwb[p], 2)
        #         return [s1, s2, torch.LongTensor([1])]
        # #随机负样本
        # else:
        #     p, n = random.sample(standard_question_index, 2)
        #     s1 = random.choice(self.knwb[p])
        #     s2 = random.choice(self.knwb[n])
        #     return [s1, s2, torch.LongTensor([-1])]
        valid_positive_keys = [k for k in self.knwb if len(self.knwb[k]) >= 2]
        # if not valid_positive_keys:
        #     raise ValueError("No valid positive samples available")

        # 随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            p = random.choice(valid_positive_keys)
            s1, s2 = random.sample(self.knwb[p], 2)
            return [s1, s2, torch.LongTensor([1])]
        # 随机负样本（保持原逻辑）
        else:
            p, n = random.sample(list(self.knwb.keys()), 2)
            s1 = random.choice(self.knwb[p])
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([-1])]

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    # def cosine_triplet_loss(self, a, p, n, margin=None):
    #     ap = self.cosine_distance(a, p)
    #     an = self.cosine_distance(a, n)
    #     if margin is None:
    #         diff = ap - an + 0.1
    #     else:
    #         diff = ap - an + margin.squeeze()
    #     return torch.mean(diff[diff.gt(0)]) #greater than


    def cosine_triplet_loss(self,a,p,n,margin=None):
        triplet_loss = nn.TripletMarginLoss(margin=1.0,p= 2,eps=1e-7)
        return  triplet_loss(a,p,n)

    #sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, target=None):
        #同时传入两个句子
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            #如果有标签，则计算loss
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            #如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    #将知识库中的问题向量化，为匹配做准备
    #每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            #将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}  #清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                test_question_vectors = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            #通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            #test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze())) #命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index] #转化成标准问编号
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = SiameseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, labels)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    main(Config)
  
# ------------------------------------------------------------------------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-
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
            question_matrixs = torch.stack(self.question_ids, dim=0)
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
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            test_question_vector = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze())) #命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index] #转化成标准问编号 
        return  self.index_to_standard_question[hit_index]

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
