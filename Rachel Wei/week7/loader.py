# -*- coding: utf-8 -*-

import pandas as pd
import json
import csv
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: 'Negative', 1: 'Positive'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    ## this is to load our specific data set.
    ## input is our JSON file
    ## output is the a list of lists. 
    ## imagine there are 1380 lines in our JSON file
    ## the output data would be a list of 1380 items
    ## each line's output has 2 elements. 
            ## first is the encoding of each characters in the title, for example [33,34,21,19, 0] for a 5 word sentence
            ## second is the type index of the text. for example, 5 for 科技 

    def load(self):
        self.data = []

        with open(self.path, encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header row 

            for row in reader:
                sensitivity = [int(row[0])]
                # sensitivity = torch.LongTensor(sensitivity)
                sensitivity = torch.tensor(sensitivity, dtype=torch.long)

                review_content = row[1]
                if self.config['model_type'] == 'bert':
                    encoded_review_content = self.tokenizer.encode(review_content
                                                                   , max_length = self.config['max_length']
                                                                   , padding = 'max_length'
                                                                   , trunction = True)
                else:
                    encoded_review_content = self.encode_sentence(review_content)
                encoded_review_content = torch.LongTensor(encoded_review_content)
                self.data.append([sensitivity, encoded_review_content])
        return

    ## set up an encoder for cases where we don't use the Bert dict
    def encode_sentence(self, text):
        input_id = []
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
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config) ## our customized data generator from above, the output of this step will be fed into DataLoader
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("data/customer_reviews.csv", Config)
    dl = load_data("data/customer_reviews.csv", Config, shuffle = True)
    df = pd.read_csv("data/customer_reviews.csv")
    for i in range(1,4):
        print(df.iloc[i])
        print(dg[i])
    
    print(Config['batch_size'])
    for i, batch in enumerate(dl):
        if i == 1:
            break
        print(batch)
        
