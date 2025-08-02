# -*- coding: utf-8 -*-

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class BertNERDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.label2id = self.load_schema(config["schema_path"])
        self.data = self.load_data(data_path)

    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
            if 'O' not in schema:  # 确保包含'O'标签
                schema['O'] = len(schema)
            return schema

    def load_data(self, data_path):
        examples = []
        with open(data_path, encoding="utf-8") as f:
            words, labels = [], []
            for line in f:
                if line.strip() == "":
                    if words:
                        examples.append((words, labels))
                        words, labels = [], []
                    continue
                parts = line.strip().split()
                words.append(parts[0])
                labels.append(parts[1] if len(parts) > 1 else "O")
            if words:
                examples.append((words, labels))
        return examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        words, labels = self.data[index]

        # Tokenize words and align labels
        tokens = ["[CLS]"]  # 添加[CLS] token
        label_ids = [self.label2id['O']]  # [CLS]位置使用'O'标签

        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # 第一个子词用原标签，其余用-1
            label_ids.extend([self.label2id[label]] + [-1] * (len(word_tokens) - 1))

        tokens.append("[SEP]")  # 添加[SEP] token
        label_ids.append(self.label2id['O'])  # [SEP]位置使用'O'标签

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.config["max_length"] - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            label_ids = label_ids + [-1] * padding_length
        else:
            input_ids = input_ids[:self.config["max_length"]]
            attention_mask = attention_mask[:self.config["max_length"]]
            label_ids = label_ids[:self.config["max_length"]]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "raw_words": words,
            "raw_labels": labels
        }

        assert len(input_ids) == len(label_ids), "输入与标签长度不一致"
        assert all(l != -1 for l in label_ids[:5]), "前5个标签包含过多-1"
        print("样例检查 - 输入:", input_ids[:10])
        print("对应标签:", label_ids[:10])  # 确认[CLS]标签不为-1


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "raw_words": [x["raw_words"] for x in batch],
        "raw_labels": [x["raw_labels"] for x in batch]
    }


def load_data(data_path, config, shuffle=True):
    dataset = BertNERDataset(data_path, config)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn
    )
