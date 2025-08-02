# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class BertNERDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.label2id = self.load_schema(config["schema_path"])
        self.max_len = config["max_length"]
        self.data = self.load_data(data_path)

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def load_data(self, path):
        data = []
        with open(path, encoding="utf8") as f:
            segments = f.read().strip().split("\n\n")
            for segment in segments:
                words = []
                labels = []
                for line in segment.strip().split("\n"):
                    if not line.strip():
                        continue
                    word, label = line.strip().split()
                    words.append(word)
                    labels.append(label)

                # 对字符级别的数据使用 BERT tokenizer
                tokens = []
                label_ids = []

                for word, label in zip(words, labels):
                    word_tokens = self.tokenizer.tokenize(word)
                    if not word_tokens:
                        word_tokens = [self.tokenizer.unk_token]
                    tokens.extend(word_tokens)

                    # 第一个子词保留标签，其余设为 -100（loss 忽略）
                    label_id = self.label2id[label]
                    label_ids.extend([label_id] + [-100] * (len(word_tokens) - 1))

                # 添加 special tokens
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                label_ids = [-100] + label_ids + [-100]

                # 转为 input_ids
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                attention_mask = [1] * len(input_ids)

                # Padding
                padding_length = self.max_len - len(input_ids)
                if padding_length > 0:
                    input_ids += [self.tokenizer.pad_token_id] * padding_length
                    attention_mask += [0] * padding_length
                    label_ids += [-100] * padding_length
                else:
                    input_ids = input_ids[:self.max_len]
                    attention_mask = attention_mask[:self.max_len]
                    label_ids = label_ids[:self.max_len]

                data.append({
                    "input_ids": torch.LongTensor(input_ids),
                    "attention_mask": torch.LongTensor(attention_mask),
                    "labels": torch.LongTensor(label_ids)
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# DataLoader 包装
def load_data(data_path, config, shuffle=True):
    dataset = BertNERDataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader
