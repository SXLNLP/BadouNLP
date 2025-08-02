# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        df = pd.read_csv(data_path)
        unique_labels = sorted(df['label'].unique())
        self.index_to_label = {i: label for i, label in enumerate(unique_labels)}
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}

        self.config["class_num"] = len(self.index_to_label)

        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        df = pd.read_csv(self.path)

        for _, row in df.iterrows():
            text = row['review']
            label = row['label']
            label_index = self.label_to_index[label]

            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(
                    text,
                    max_length=self.config["max_length"],
                    padding='max_length',
                    truncation=True
                )
            else:
                input_id = self.encode_sentence(text)

            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label_index])
            self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

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
            token_dict[token] = index + 1
    token_dict["[UNK]"] = len(token_dict) + 1
    token_dict["[PAD]"] = 0
    return token_dict


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl