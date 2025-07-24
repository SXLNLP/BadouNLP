


"""
数据加载
"""
import csv
import json

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataGenerator:
    def __init__(self,data_path,config):
        self.config = config
        self.path = data_path
        self.index_to_label = {1: '好评', 0: '差评'}
        # self.label_to_index = dict((v,k) for k,v in self.index_to_label.items())
        # self.label_to_index = {v:k for k,v in self.index_to_label.items()}
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            # 加载tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()
    def load(self):
        # 加载数据
        self.data = []
        with open(self.path,encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                label = int(line[0])
                comment = line[1]
                if self.config["model_type"] == "bert":
                    # 关键修改：使用tokenizer()获取包含attention_mask的字典
                    input_id = self.tokenizer.encode(
                        comment,
                        max_length=self.config["max_length"],
                        padding="max_length",
                        truncation=True
                    )
                else:
                    # 非BERT模型的处理（手动生成attention_mask）
                    input_id = self.encode_sentence(comment)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id,label_index])
        return

    # 输入转变为词表的索引
    def encode_sentence(self,text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id
    # 补全或截断输入，保证输入为定长
    def padding(self,input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0]*(self.config["max_length"]-len(input_id))
        return input_id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

# 加载词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path,encoding="utf-8") as f:
        for index,line in enumerate(f):
            token = line.strip()
            token_dict[token] = index+1 #0留给padding位置，所以从1开始
    return token_dict

# torch自带的dataloader类作为封装
def load_data(data_path,config,shuffle=True):
    dg = DataGenerator(data_path,config)
    dl = DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return dl
