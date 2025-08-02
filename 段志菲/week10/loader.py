# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        
        # 使用BERT的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.config["vocab_size"] = self.tokenizer.vocab_size
        
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    def prepare_data(self, title, content):
        # 编码输入和输出
        inputs = self.tokenizer(
            content,
            max_length=self.config["input_max_length"],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            title,
            max_length=self.config["output_max_length"],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.data.append([
            inputs['input_ids'].squeeze(0),
            inputs['attention_mask'].squeeze(0),
            targets['input_ids'].squeeze(0)
        ])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
