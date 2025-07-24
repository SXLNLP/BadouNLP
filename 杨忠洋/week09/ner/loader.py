# -*- coding: utf-8 -*-

import json

import jieba
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.data = None
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                if self.config["use_bert"]:
                    self.bert_encode_sentence(sentence, labels)
                else:
                    input_ids = self.encode_sentence(sentence)
                    labels = self.padding(labels, -1)
                    self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
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

    def bert_encode_sentence(self, sentence, labels):
        tokenizer = BertTokenizerFast.from_pretrained(self.config["bert_path"])
        encode_words = tokenizer.encode_plus(text=sentence,
                                             padding="max_length",
                                             max_length=self.config["max_length"],
                                             truncation=True,
                                             return_tensors="pt",
                                             is_split_into_words=True)
        labels_sync = []
        previous_word_id = None
        for word_id in encode_words.word_ids():
            if word_id is None:
                labels_sync.append(-100)
            elif word_id != previous_word_id:
                labels_sync.append(labels[word_id] if (0 > labels[word_id] >= self.config["class_num"]) else -100)
            elif word_id == previous_word_id:
                labels_sync.append(labels[word_id] if (0 > labels[word_id] >= self.config["class_num"]) else -100)
            previous_word_id = word_id
        self.data.append({
            "input_id": encode_words["input_ids"].squeeze(0),
            "attention_mask": encode_words["attention_mask"].squeeze(0),
            "labels": torch.LongTensor(labels_sync)
        })
        return


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
