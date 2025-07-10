# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import csv
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class GlobalDataManager:
    _instance = None #单例模式

    def __new__(cls,*args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self, config):
        self.config = config
        self.vocab = load_vocab(config["vocab_path"]) 
        self.load_encode_and_split()       
        self.config["vocab_size"] = len(self.vocab)



    def load_encode_and_split(self):
        with open(self.config["data_path"], 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 跳过CSV文件的标题行（如果存在）
            header = next(reader) # 如果没有标题行，可以注释掉这行
            data = list(reader) # 将所有行读取到列表中
            data = self.encode(data,self.config["model_type"]=="bert") #将每一行的句子编码为 id 序列
            #print(data)
        # shuffle
        np.random.shuffle(data) # 对数据列表进行打乱

        # shuffle后拆分数据为训练集和测试集（50%为训练集，25%为测试集，25%为验证集）
        train_size = int(len(data) * 0.5)
        test_size = int(len(data) * 0.25)
        train_data = data[:train_size]
        test_data = data[train_size:train_size+test_size]
        val_data = data[train_size+test_size:]


        self.train_data = DataLoader(train_data, batch_size=self.config["batch_size"], shuffle=False)
        self.test_data = DataLoader(test_data, batch_size=self.config["batch_size"], shuffle=False)
        self.val_data = DataLoader(val_data, batch_size=self.config["batch_size"], shuffle=False)
        

        # # 检查DataLoader中第一个批次的形状
        # # 对于训练数据
        # if len(self.train_data) > 0:
        #     first_batch = next(iter(self.train_data))
        #     # 假设first_batch是一个包含[input_ids, labels]的列表或元组
        #     print(f"训练数据第一个批次中输入Tensor的形状: {first_batch[0].shape}")
        #     print(f"训练数据第一个批次中标签Tensor的形状: {first_batch[1].shape}")
        
        # # 对于测试数据
        # if len(self.test_data) > 0:
        #     first_batch = next(iter(self.test_data))
        #     print(f"测试数据第一个批次中输入Tensor的形状: {first_batch[0].shape}")
        #     print(f"测试数据第一个批次中标签Tensor的形状: {first_batch[1].shape}")
        
        # # 对于验证数据
        # if len(self.val_data) > 0:
        #     first_batch = next(iter(self.val_data))
        #     print(f"验证数据第一个批次中输入Tensor的形状: {first_batch[0].shape}")
        #     print(f"验证数据第一个批次中标签Tensor的形状: {first_batch[1].shape}")

    

    def encode(self,data,is_bert):
        data_new = []
        if is_bert:
            tokenizer = BertTokenizer.from_pretrained(r"bert-base-chinese")
            for index, line in enumerate(data):
                # print(f"index={index}")
                # print(f"line={line}")
                type_value =data[index][0]
                # print(f"type_value={type_value}")
                input_id = torch.LongTensor(tokenizer.encode(line[1], max_length=self.config["max_length"], pad_to_max_length=True))
                # print(f"input_id={input_id}")
                #type = int(type)
                label_index = torch.LongTensor([int(type_value)])
                # print(f"label_index={label_index}")
                data_new.append([input_id, label_index])
        else:
            # 如果不是 bert模型，那就采用字表的方式来将句子转换为索引/id 序列
            for index, line in enumerate(data):
                #print(line[1])
                type_value=data[index][0]
                input_id = torch.LongTensor(self.encode_sentence(line[1]))
                label_index = torch.LongTensor([int(type_value)])
                data_new.append([input_id, label_index])
        return data_new
        # data[:,0] = torch.LongTensor(data[:,0])
        # data[:,1] = torch.LongTensor(data[:,1])
        # print(data)
        #data = torch.LongTensor(data)
    
    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]] #截断
        input_id += [0] * (self.config["max_length"] - len(input_id)) #补齐
        return input_id
    
        #print(data)
    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id  

def load_vocab(vocab_path):
    """词表是一个字典，键为词，值为id"""
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict



if __name__ == "__main__":
    from config import Config
    gdm = GlobalDataManager(Config)
    # print(gdm.train_data[:10])
    # print(gdm.test_data[:10])
    # print(gdm.val_data[:10])
    # print("---训练集-----")
    # print(gdm.train_data[:10])
    # print("---测试集-----")
    # print(gdm.test_data[:10])
    