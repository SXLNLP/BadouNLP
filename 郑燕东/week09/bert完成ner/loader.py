import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
"""
数据加载
"""


class DataGenerator: #初始化
    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"],truncation=True,padding="max_length",max_length=config["max_length"])
        self.path = data_path #原始数据文件路径
        self.vocab = load_vocab(config["vocab_path"]) #词表路径
        self.config["vocab_size"] = len(self.vocab) #记录词表大小
        self.sentences = []
        self.schema = self._load_schema(config["schema_path"]) #标签路径
        self.load()

    def _load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def load(self):
        self.data = [] #存储数值化后的训练数据
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n") #分割文档为多个句子段（segment）
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":   #一个字对应一个label 用空格切分
                        continue
                    char, label = line.split()  #前面是字 后面是label
                    sentenece.append(char)
                    labels.append(self.schema[label])

                self.sentences.append("".join(sentenece))  #字扔到sentences里 保存原始字符串
                encoded = self.tokenizer(" ".join(sentenece), add_special_tokens=True,return_tensors="pt",truncation=True,
                padding='max_length',max_length=self.config["max_length"])
                # input_ids = self.encode_sentence(sentenece)  #对输入的文本进行编码 转化成torch的tensor
                labels = self.padding(labels, -1)   #label扔到label里
                aligned_labels = []
                char_ptr = 0
                for word_id in encoded.word_ids():
                    if word_id is None:
                        aligned_labels.append(-100)
                    elif word_id >= len(labels):
                        aligned_labels.append(labels[-1])
                    else:
                        aligned_labels.append(labels[word_id])
                # labels = self.align_labels(sentenece,labels,input_ids)
                # self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
                self.data.append((encoded["input_ids"].squeeze(0),encoded["attention_mask"].squeeze(0),torch.LongTensor(aligned_labels)))
        return

    def encode_sentence(self, text, padding=True): #将文本转换为词/字ID序列并进行填充对齐
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding: #执行padding()方法进行序列长度标准化
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):  #多个样本时该padding，该截断截断
        input_id = input_id[:self.config["max_length"]] #通过切片操作保留前max_length个元素（截断处理）
        input_id += [pad_token] * (self.config["max_length"] - len(input_id)) #填充处理：在序列末尾补充pad_token至目标长度
        return input_id

class NERDataset(Dataset):
    def __init__(self, data_generator,config=None):
        super().__init__()
        self.data = data_generator.data
        self.config = config if config else {}
        self.sentences = data_generator.sentences  # 原始文本
        self.labels = [item[2] for item in data_generator.data]
        self.dataset = self._build_dataset(data_generator)

    def _build_dataset(self, data_generator):
        """构建结构化数据集"""
        return [
            {
                "raw_text": sent,
                "input_ids": encoded[0],
                "attention_mask": encoded[1],
                "labels": encoded[2],
                "tokenized_text": data_generator.tokenizer.tokenize(sent)
            }
            for sent, encoded in zip(data_generator.sentences, data_generator.data)
        ]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return {
            "input_ids": self.data[index][0].cuda(),
            "attention_mask": self.data[index][1].cuda(),
            "labels": self.data[index][2].cuda()
        }

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f) #自动处理JSON到Python对象的转换

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f: #读取UTF-8编码的词汇表文件（每行一个token）
        for index, line in enumerate(f):
            token = line.strip() #去除每行首尾空白字符
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True): #shuffle：控制是否打乱数据顺序，默认启用
    dg = DataGenerator(data_path, config)
    # dataset = NERDataset(dg, config)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=dg.tokenizer,
        padding=True,
        max_length=config["max_length"],label_pad_token_id=-100
    )
    # dataset = NERDataset(data_path, config)
    # dl = DataLoader(NERDataset(dg, config), batch_size=config["batch_size"], collate_fn=data_collator,shuffle=shuffle) #从config字典获取批次大小
    dataset = NERDataset(dg,config)
    return dataset



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner/ner_data/train", Config)

