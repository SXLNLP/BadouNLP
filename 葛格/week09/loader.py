# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])

        if config.get("use_bert", False):
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        else:
            self.vocab = load_vocab(config["vocab_path"])
            config["vocab_size"] = len(self.vocab)

        self.load()

    def load(self):
        self.data = []
        self.sentences = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().strip().split("\n\n")
            for segment in segments:
                if not segment.strip():
                    continue
                sentence = []
                label_list = []
                for line in segment.strip().split("\n"):
                    char, label = line.strip().split()
                    sentence.append(char)
                    label_list.append(self.schema[label])

                self.sentences.append("".join(sentence))  # 原始句子
                input_ids, labels, attention_mask = self.encode_sentence(sentence, label_list)

                self.data.append((input_ids, labels, attention_mask))

    def encode_sentence(self, chars, labels):
        tokens = []
        label_ids = []

        for char, label in zip(chars, labels):
            tokenized = self.tokenizer.tokenize(char)
            if not tokenized:
                tokenized = [self.tokenizer.unk_token]

            tokens.extend(tokenized)
            # 只保留第一个 subword 的标签，后续 subword 设为 -100（忽略）
            label_ids.extend([label] + [-100] * (len(tokenized) - 1))

        # 添加特殊符号 [CLS], [SEP]
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        label_ids = [-100] + label_ids + [-100]

        encoding = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoding)

        encoding = self.padding(encoding, pad_token=self.tokenizer.pad_token_id)
        label_ids = self.padding(label_ids, pad_token=-100)
        attention_mask = self.padding(attention_mask, pad_token=0)

        return torch.LongTensor(encoding), torch.LongTensor(label_ids), torch.LongTensor(attention_mask)

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, ids, pad_token=0):
        ids = ids[:self.config["max_length"]]
        ids += [pad_token] * (self.config["max_length"] - len(ids))
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, labels, attention_mask = self.data[index]
        if len(self.sentences) > index:
            sentence = self.sentences[index]
        else:
            sentence = ""
        return input_ids, labels, attention_mask, sentence

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("ner_data/train", Config)
    print(type(dg))
    # vocab_path = "chars.txt"
    # vocab = load_vocab(vocab_path)
    # print(vocab)
