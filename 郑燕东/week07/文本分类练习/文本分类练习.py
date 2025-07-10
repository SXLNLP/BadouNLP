Config = {
    "model_path": r"E:\09-python\homework\week7",
    "train_data_path": r"E:\09-python\homework\week7\文本分类练习.csv",
    "valid_data_path": r"E:\09-python\homework\week7\文本分类练习.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",   #通过这个字段切换不同的模型结构
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\09-python\04-八斗课件\第六周 语言模型\bert-base-chinese",
    "seed": 987
}
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":   #bert也可以单独作为pooling使用接63行作为encoding
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False) #强制返回元组而非字典格式的输出
            hidden_size = self.encoder.config.hidden_size #动态获取BERT的隐层维度（如BERT-base为768），确保后续层参数匹配
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            #sequence_output:batch_size, max_len, hidden_size
            #pooler_output:batch_size, hidden_size
            x = self.encoder(x)           #把bert作为encoder之后接70行
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim)

        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        #可以采用pooling的方式得到句向量
        if self.pooling_style == "max":  #可作为config文件的一步，config.py文件里
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        #pooling以后作转置
        #也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]
        predict = self.classify(x)   #input shape:(batch_size, input_dim)  再过线性层作分类
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        #仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)               #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1) #二层线性
            x = self.bn_after_ff[i](x + l2)        #残差后过bn
        return x


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False) #先用官方的库把bert引进来
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]  #把bert输出的结果传入162行的lstm里，也可以是rnn，cnn 170行
        x, _ = self.rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x

class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()  #bert中间层的方式
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True #结果为True时会把中间结果输出来见下187行

    def forward(self, x):
        layer_states = self.bert(x)[2]#(13, batch, len, hidden)
        layer_states = torch.add(layer_states[-2], layer_states[-1]) #[-2]取的最后两层对应的第4种方法
        return layer_states


#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    # Config["class_num"] = 3
    # Config["vocab_size"] = 20
    # Config["max_length"] = 5
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))


    # model = TorchModel(Config)
    # label = torch.LongTensor([1,2])
    # print(model(x, label))
  # -*- coding: utf-8 -*-
import csv
import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
"""
数据加载（每次需要修改）
"""


class DataGenerator:
    def __init__(self, data_path, config,is_train=True,indices=None):
        self.config = config #配置字典
        self.path = data_path #存储数据路径
        self.is_train = is_train
        self.indices = indices
        self.index_to_label = {0: '差评',1:'好评'} #建立标签映射关系
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label) #设置类别数量到配置中
        if self.config["model_type"] == "bert": #当模型类型为BERT时，加载预训练分词器BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])#要用bert自己的词表，下载模型地址
        self.vocab = load_vocab(config["vocab_path"]) #加载自定义词汇表（通过load_vocab函数）
        self.config["vocab_size"] = len(self.vocab) #记录词汇表大小vocab_size到配置中
        self.data=[]
        self.labels=[]
        self.load()


    def load(self): #将JSON格式的原始数据转换为模型可处理的张量格式 原始JSON → 解析标签 → 文本编码 → 张量转换 → 存储到data列表
        self.data = []
        with open(self.path, encoding="utf8") as f:
            reader = csv.DictReader(f)
            for idx,row in enumerate(reader):
                y = row['label']
                x = row['review']
                self.labels.append(y)
                if self.config["model_type"] == "bert":
                    x = self.tokenizer.encode(x, max_length=self.config["max_length"], padding = 'max_length',truncation = True,return_attention_mask=True)
                else:
                    x = self.encode_sentence(x)
                x = torch.LongTensor(x)
                y = torch.LongTensor([int(y)])
                self.data.append([x,y]) #存储到data列表
        return

    def encode_sentence(self, text): #将输入文本转换为词汇表索引序列并进行填充处理。
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"])) #字符查找失败时返回未知标记的索引
        input_id = self.padding(input_id) #对序列进行截断/填充的方法
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]] #先截断超过最大长度的部分
        input_id += [0] * (self.config["max_length"] - len(input_id)) #用0填充不足最大长度的部分
        return input_id

    def __len__(self):
        return len(self.data) #返回数据集样本总数 必需方法，供DataLoader确定迭代次数

    def __getitem__(self, index):
        return self.data[index] #按索引返回单个样本 与__len__共同构成PyTorch数据集基本接口

def load_vocab(vocab_path):  #构建词表索引
    token_dict = {} #构建token到index的映射字典
    with open(vocab_path, encoding="utf8") as f: #从指定路径读取词汇表文件 词汇表文件路径，每行一个token
        for index, line in enumerate(f):  #enumerate(f)逐行读取并自动计数
            token = line.strip() #去除每行首尾空白字符
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始 index + 1确保索引从1开始（0通常用于padding或特殊标记）
    return token_dict

def split_dataset(dataset, test_size=0.1, stratify=None, random_state=42):
    train_indices,test_indices = train_test_split(list(range(len(dataset))))
    test_size = test_size,
    stratify = stratify,
    random_state = random_state
    return train_indices,test_indices
    
def load_data(data_path,config,test_size=0.1,shuffle=True):
    #加戴数据并划分为训练集和测试集
    full_dataset = DataGenerator(data_path, config)
    #划分好数据集
    train_indices, test_indices = split_dataset(
        full_dataset,
        test_size=test_size,
        stratify=full_dataset.labels,  # 按标签分层
        random_state=config.get("random_seed", 42)
    )
    # 创建训练集和测试集
    train_dataset = DataGenerator(data_path, config, indices=train_indices)
    test_dataset = DataGenerator(data_path, config, is_train=False, indices=test_indices)
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    from config import Config
    file_path = r"E:\09-python\homework\week7\文本分类练习.csv"
    train_loader, test_loader = load_data(file_path, Config)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    print("训练集第一个样本:")
    print(train_loader.dataset[0])
# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试（验证模型的环节）
"""

class Evaluator: #用于模型评估的Evaluator类初始化方法
    def __init__(self, config, model, logger): #存储模型配置config、模型对象model和日志记录器logger
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False) #设置shuffle=False保持数据顺序
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果 初始化统计字典stats_dict记录正确/错误预测数

    def eval(self, epoch): #这是一个模型评估方法，用于在验证集上测试模型性能
        self.logger.info("开始测试第%d轮模型效果：" % epoch) #epoch：当前训练轮次，用于日志记录
        self.model.eval() #将模型设为评估模式（关闭dropout等）
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad(): #禁用自动求导提升效率
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results) #统计方法：通过write_stats和show_stats实现（需另行定义）体现其数据记录职责
        acc = self.show_stats() #show_stats方法形成完整评估流程
        return acc #模型在验证集上的准确率

    def write_stats(self, labels, pred_results): #这是一个用于模型评估的统计记录方法，主要功能是比对预测结果和真实标签并更新统计字典
        assert len(labels) == len(pred_results)  #使用assert确保标签和预测结果数量一致
        for true_label, pred_label in zip(labels, pred_results): #通过zip实现逐样本比对
            pred_label = torch.argmax(pred_label) #torch.argmax获取预测类别（适用于多分类)
            if int(true_label) == int(pred_label): #强制转换为int类型确保比较一致性
                self.stats_dict["correct"] += 1 #统计更新 正确预测时递增stats_dict["correct"]
            else:
                self.stats_dict["wrong"] += 1 #统计更新 错误预测时递增stats_dict["wrong"]
        return #无返回值，直接修改实例变量stats_dict

    def show_stats(self): #主要功能是计算并输出模型在验证集上的性能指标
        correct = self.stats_dict["correct"] #从统计字典中提取正确/错误计数
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载（每次需要修改）
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config #配置字典
        self.path = data_path #存储数据路径
        self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'} #建立标签映射关系
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label) #设置类别数量到配置中
        if self.config["model_type"] == "bert": #当模型类型为BERT时，加载预训练分词器BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])#要用bert自己的词表，下载模型地址
        self.vocab = load_vocab(config["vocab_path"]) #加载自定义词汇表（通过load_vocab函数）
        self.config["vocab_size"] = len(self.vocab) #记录词汇表大小vocab_size到配置中
        self.load()


    def load(self): #将JSON格式的原始数据转换为模型可处理的张量格式 原始JSON → 解析标签 → 文本编码 → 张量转换 → 存储到data列表
        self.data = []
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line) #解析每行JSON数据
                tag = line["tag"]
                label = self.label_to_index[tag] #将标签文本映射为数字索引
                title = line["title"]
                if self.config["model_type"] == "bert":#有个encode作编码，就不用再补0了
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                    # BERT模型：使用tokenizer自动处理文本 自定义模型：调用encode_sentence方法
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index]) #存储到data列表
        return

    def encode_sentence(self, text): #将输入文本转换为词汇表索引序列并进行填充处理。
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"])) #字符查找失败时返回未知标记的索引
        input_id = self.padding(input_id) #对序列进行截断/填充的方法
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]] #先截断超过最大长度的部分
        input_id += [0] * (self.config["max_length"] - len(input_id)) #用0填充不足最大长度的部分
        return input_id

    def __len__(self):
        return len(self.data) #返回数据集样本总数，供DataLoader确定迭代次数

    def __getitem__(self, index): #按索引返回单个样本 与__len__共同构成PyTorch数据集基本接口
        return self.data[index]

def load_vocab(vocab_path): #构建词表索引
    token_dict = {} #构建token到index的映射字典
    with open(vocab_path, encoding="utf8") as f: #从指定路径读取词汇表文件 词汇表文件路径，每行一个token
        for index, line in enumerate(f):  #enumerate(f)逐行读取并自动计数
            token = line.strip() #去除每行首尾空白字符
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始 index + 1确保索引从1开始（0通常用于padding或特殊标记）
    return token_dict


#用torch自带的DataLoader类封装数据 通过DataGenerator和DataLoader构建数据加载管道。
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
