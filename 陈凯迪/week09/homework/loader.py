# -*- coding: utf-8 -*-

import json

import jieba
import torch
from torch.utils.data import DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                tokens = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    tokens.append(char)
                    labels.append(self.schema[label])

                sentence = "".join(tokens)
                self.sentences.append(sentence)

                # BERT分词
                encoding = self.tokenizer.encode_plus(
                    tokens,
                    is_split_into_words=True,
                    max_length=self.config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                # 对齐标签
                word_ids = encoding.word_ids()
                aligned_labels = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:  # 特殊token [CLS], [SEP]
                        aligned_labels.append(-100)
                    elif word_idx != previous_word_idx:  # 单词的第一个token
                        aligned_labels.append(labels[word_idx])
                    else:  # 同一个单词的后续token
                        aligned_labels.append(-100)
                    previous_word_idx = word_idx

                self.data.append([
                    encoding["input_ids"].squeeze(0),
                    encoding["attention_mask"].squeeze(0),
                    torch.LongTensor(aligned_labels)
                ])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

