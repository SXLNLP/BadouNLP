# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

class DataGenerator(Dataset):
    def __init__(self, data_path, config, tokenizer=None):
        self.config = config

        self.tokenizer = BertTokenizerFast.from_pretrained(
            config["bert_path"], use_fast=True)
        self.schema = json.load(open(config["schema_path"], encoding="utf8"))
        self.label2id = self.schema
        self.data = self.load(data_path)

    def load(self, path):
        data = []
        with open(path, encoding="utf8") as f:
            for seg in f.read().split("\n\n"):
                chars, labels = [], []
                for line in seg.strip().split("\n"):
                    if not line:
                        continue
                    c, l = line.split()
                    chars.append(c)
                    labels.append(self.label2id[l])

                # 用 BERT tokenizer 分字
                encoding = self.tokenizer(
                    chars,
                    is_split_into_words=True,
                    max_length=self.config["max_length"],
                    padding='max_length',
                    truncation=True
                )
                input_ids = encoding["input_ids"]
                # 把 label 对齐到 sub-word 级别
                aligned_labels = []
                word_ids = encoding.word_ids()
                for w in word_ids:
                    if w is None:
                        aligned_labels.append(-1)
                    else:
                        aligned_labels.append(labels[w])
                data.append([torch.LongTensor(input_ids),
                             torch.LongTensor(aligned_labels)])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, shuffle=True):
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], use_fast=True)
    ds = DataGenerator(data_path, config, tokenizer)
    return DataLoader(ds,
                      batch_size=config["batch_size"],
                      shuffle=shuffle,
                      drop_last=False)
