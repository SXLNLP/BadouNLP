# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        self.config["mask_idx"] = self.vocab["[MASK]"]
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    #文本到对应的index
    #头尾分别加入[cls]和[sep]
    def encode_sentence(self, text, max_length, with_cls_token=True, with_sep_token=True):
        input_id = []
        if with_cls_token:
            input_id.append(self.vocab["[CLS]"])
        for char in text:
            if char == self.vocab["[MASK]"]:
                input_id.append(self.vocab["[MASK]"])
            else:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if with_sep_token:
            input_id.append(self.vocab["[SEP]"])
        input_id = self.padding(input_id, max_length)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id
    
    def mask_sentence(self, content):
        """
        对输入的句子进行mask处理，使其成为预测下一个token的任务。
        具体是将content中的每一个字符依次mask掉，然后使用前面的字符预测后面被mask掉的字符
        """
        masked_data = []
        l = len(content)
        content_list = list(content)
        for i in range(l):
            mask_seq = content_list[:i] + [self.vocab["[MASK]"]] * (l - i)
            masked_data.append(mask_seq)
        return masked_data


    #输入输出转化成序列
    def prepare_data(self, title, content):
        l = len(content)
       #input_seq = self.encode_sentence(content, self.config["input_max_length"], False, False) #输入序列
       #output_seq = self.encode_sentence(title, self.config["output_max_length"], True, False) #输出序列
        masked_seqs = self.mask_sentence(content)
        #print(f"masked_seqs: {len(masked_seqs)}")

        #gold = self.encode_sentence(title, self.config["output_max_length"], False, True) #不进入模型，用于计算loss
        for i in range(1,l):
            input_seq = masked_seqs[i-1]
            output_seq = masked_seqs[i]
            
            input_seq = self.encode_sentence(input_seq, self.config["input_max_length"])
            output_seq = self.encode_sentence(output_seq, self.config["output_max_length"])
            gold = self.encode_sentence(output_seq, self.config["output_max_length"])
            self.data.append([torch.LongTensor(input_seq),
                            torch.LongTensor(output_seq),
                            torch.LongTensor(gold)])

        return


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    print(f"数据加载完成，数据量：{len(dg)}")
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_data_path"], Config, 1)
    #print([(i, n) for (i, n) in enumerate(dl) if i ==0])

