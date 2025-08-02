# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NERDataset(Dataset):
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"], add_special_tokens=True)
        self.label2id = config["label2id"]
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.data = self.load_data(data_path)

    def load_data(self, path):
        data = []
        with open(path, encoding="utf8") as f:
            words, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        data.append((words, labels))
                        words, labels = [], []
                    continue
                if " " in line:
                    splits = line.split()
                else:
                    splits = line.split("\t")
                if len(splits) < 2:
                    continue
                words.append(splits[0])
                labels.append(splits[1])
            if words:
                data.append((words, labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, labels = self.data[idx]
        tokens = []
        label_ids = []
        tokens.append("[CLS]")
        label_ids.append(-100)
        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = ["[UNK]"]
            tokens.extend(word_tokens)
            # 只对第一个子词打原始标签，其余为-100
            label_ids.append(self.label2id.get(label, self.label2id["O"]))
            for _ in word_tokens[1:]:
                label_ids.append(-100)
        tokens.append("[SEP]")
        label_ids.append(-100)
        # 转为input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        # padding
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            label_ids += [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label_ids = label_ids[:self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

def load_data(data_path, config, logger, shuffle=True):
    dataset = NERDataset(data_path, config, logger)
    return dataset


if __name__ == "__main__":
    from config import Config
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


