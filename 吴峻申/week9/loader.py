import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


class NerDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.data = []
        self.load(data_path)

    def load(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n\n')
            for block in lines:
                if not block.strip():
                    continue

                tokens, labels = [], []
                for line in block.split('\n'):
                    if line.strip() == '':
                        continue
                    char, label = line.split()
                    tokens.append(char)
                    labels.append(self.schema[label])

                self.sentences.append(''.join(tokens))

                # 使用分词器处理文本
                inputs = self.tokenizer(
                    tokens,
                    padding='max_length',
                    max_length=self.config["max_length"],
                    truncation=True,
                    is_split_into_words=True,
                    return_tensors='pt'
                )

                input_ids = inputs['input_ids'].squeeze(0)
                attention_mask = inputs['attention_mask'].squeeze(0)

                # 对齐标签
                word_ids = inputs.word_ids()
                aligned_labels = []
                previous_word_idx = None

                for word_idx in word_ids:
                    if word_idx is None:
                        # 特殊token
                        aligned_labels.append(-100)
                    elif word_idx != previous_word_idx:
                        # 新单词的第一个token
                        aligned_labels.append(labels[word_idx])
                    else:
                        # 同一个单词的子词
                        aligned_labels.append(-100)
                    previous_word_idx = word_idx

                # 确保标签长度匹配
                aligned_labels = aligned_labels[:self.config["max_length"]]
                while len(aligned_labels) < self.config["max_length"]:
                    aligned_labels.append(-100)

                label_ids = torch.tensor(aligned_labels)

                self.data.append((input_ids, attention_mask, label_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load_schema(schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return {k: int(v) for k, v in schema.items()}


def load_data(data_path, config, shuffle=True):
    dataset = NerDataset(data_path, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch])
        )
    )
    return dataloader
