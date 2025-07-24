# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载，使用BertTokenizer实现标签对齐
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
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

                # 标签对齐
                encoded_result = self.encode_sentence_with_labels(sentence, labels)
                if encoded_result is not None:
                    input_ids, attention_mask, aligned_labels = encoded_result
                    self.data.append([
                        torch.LongTensor(input_ids),
                        torch.LongTensor(attention_mask),
                        torch.LongTensor(aligned_labels)
                    ])
        return

    def encode_sentence_with_labels(self, chars, labels):
        """
        使用BertTokenizer实现标签对齐
        通过字符串匹配，建立token到字符位置的映射
        """
        # 将字符连接成句子
        sentence = "".join(chars)

        # 首先tokenize获取tokens
        tokens = self.tokenizer.tokenize(sentence)

        # 截断到最大长度-2（为CLS和SEP留空间）
        max_seq_len = self.config["max_length"] - 2
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        # 对齐标签
        aligned_labels = self.align_labels_with_chars(tokens, chars, labels)

        # 添加特殊tokens
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        aligned_labels = [-1] + aligned_labels + [-1]  # CLS和SEP标签为-1

        # 转换为input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 创建attention_mask
        attention_mask = [1] * len(input_ids)

        # padding到最大长度
        while len(input_ids) < self.config["max_length"]:
            input_ids.append(0)  # [PAD] token id
            attention_mask.append(0)
            aligned_labels.append(-1)

        # 截断到最大长度
        input_ids = input_ids[:self.config["max_length"]]
        attention_mask = attention_mask[:self.config["max_length"]]
        aligned_labels = aligned_labels[:self.config["max_length"]]

        return input_ids, attention_mask, aligned_labels

    def align_labels_with_chars(self, tokens, chars, labels):
        """
        实现token到字符的标签对齐
        对于每个token，找到其在原始文本中的起始字符位置，使用该位置的标签
        """
        aligned_labels = []
        char_pointer = 0  # 指向当前处理到的字符位置
        original_text = "".join(chars)

        for token in tokens:
            if char_pointer >= len(chars):
                # 超出字符范围，标记为-1
                aligned_labels.append(-1)
                continue

            # 处理subword（以##开头的token）
            if token.startswith("##"):
                # 去掉##前缀
                token_text = token[2:]
                # 继续在当前位置匹配
                if char_pointer < len(chars):
                    aligned_labels.append(labels[char_pointer])
                    # 移动字符
                    char_pointer += len(token_text)
                else:
                    aligned_labels.append(-1)
            else:
                # 正常token，在原文中查找匹配位置
                token_text = token

                # 在原文中从当前位置开始查找token
                found = False
                search_start = char_pointer

                # 在一定范围内搜索匹配的位置
                for search_pos in range(search_start, min(search_start + 5, len(chars))):
                    # 检查从search_pos开始是否能匹配token_text
                    if self.match_token_at_position(original_text, search_pos, token_text):
                        # 找到匹配位置，使用该位置的标签
                        aligned_labels.append(labels[search_pos])
                        char_pointer = search_pos + len(token_text)
                        found = True
                        break

                if not found:
                    # 如果没找到，使用当前位置的标签
                    if char_pointer < len(chars):
                        aligned_labels.append(labels[char_pointer])
                        char_pointer += 1
                    else:
                        aligned_labels.append(-1)

        return aligned_labels

    def match_token_at_position(self, text, start_pos, token_text):
        """
        检查在指定位置是否能匹配token文本
        """
        if start_pos + len(token_text) > len(text):
            return False

        # 直接字符串匹配
        return text[start_pos:start_pos + len(token_text)] == token_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("./ner_data/train", Config)