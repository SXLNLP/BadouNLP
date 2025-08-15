# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                text = line["text"]
                label = line["label"]
                encoding = self.tokenizer.encode_plus(
                    text,
                    max_length=self.config["max_length"],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_id = encoding['input_ids'].squeeze(0)
                label_index = torch.tensor(label, dtype=torch.long)
                self.data.append((input_id, label_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    return DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
