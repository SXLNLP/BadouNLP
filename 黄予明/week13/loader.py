# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer
"""
NER序列标注数据加载
"""


class NERDataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        
        # 从schema.json加载标签映射
        if "schema_path" in config and os.path.exists(config["schema_path"]):
            import json
            with open(config["schema_path"], "r", encoding="utf8") as f:
                self.label_to_index = json.load(f)
            print(f"✅ 从schema.json加载标签映射: {len(self.label_to_index)}个标签")
        else:
            # 默认BIO格式标签映射
            self.label_to_index = {
                'O': 0,           # Outside
                'B-PER': 1,       # Begin-Person
                'I-PER': 2,       # Inside-Person  
                'B-LOC': 3,       # Begin-Location
                'I-LOC': 4,       # Inside-Location
                'B-ORG': 5,       # Begin-Organization
                'I-ORG': 6,       # Inside-Organization
                'B-MISC': 7,      # Begin-Miscellaneous
                'I-MISC': 8,      # Inside-Miscellaneous
            }
            print("⚠️  未找到schema.json，使用默认BIO标签映射")
        
        # 添加填充标签
        self.label_to_index['[PAD]'] = -100
        
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.config["class_num"] = len(self.label_to_index) - 1  # 不包含PAD标签
        
        # 初始化tokenizer（使用Fast版本支持word_ids）
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizerFast.from_pretrained(
                config["pretrain_model_path"], 
                local_files_only=True
            )
        elif self.config["model_type"] == "deepseek":
            self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"])
            # 为DeepSeek tokenizer设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab = load_vocab(config["vocab_path"]) if os.path.exists(config["vocab_path"]) else {}
        self.config["vocab_size"] = len(self.vocab) if self.vocab else self.tokenizer.vocab_size
        self.load()

    def load(self):
        self.data = []
        
        # 检查是否是目录还是文件
        if os.path.isdir(self.path):
            # 如果是目录，读取所有.txt文件
            for filename in os.listdir(self.path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.path, filename)
                    self._load_conll_file(file_path)
        else:
            # 如果是单个文件
            self._load_conll_file(self.path)
            
    def _load_conll_file(self, file_path):
        """加载CoNLL格式的NER数据"""
        with open(file_path, encoding="utf8") as f:
            sentences = []
            labels = []
            current_tokens = []
            current_labels = []
            
            for line in f:
                line = line.strip()
                if line == "":  # 空行表示句子结束
                    if current_tokens:
                        sentences.append(current_tokens)
                        labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                elif line.startswith("-DOCSTART-"):  # 文档开始标记
                    continue
                else:
                    # 处理 token\tlabel 格式
                    parts = line.split('\t') if '\t' in line else line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[-1]  # 最后一列是标签
                        current_tokens.append(token)
                        current_labels.append(label)
            
            # 处理最后一个句子
            if current_tokens:
                sentences.append(current_tokens)
                labels.append(current_labels)
        
        # 转换为模型输入格式
        for tokens, token_labels in zip(sentences, labels):
            input_ids, label_ids = self._tokenize_and_align_labels(tokens, token_labels)
            if input_ids is not None and label_ids is not None:
                self.data.append([input_ids, label_ids])
    
    def _tokenize_and_align_labels(self, tokens, labels):
        """tokenize并对齐标签"""
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized_inputs["input_ids"].squeeze(0)
        
        # 对齐标签
        word_ids = tokenized_inputs.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊token ([CLS], [SEP], [PAD]) 设置为-100
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # 新词的第一个subword使用原始标签
                if word_idx < len(labels):
                    label = labels[word_idx]
                    aligned_labels.append(self.label_to_index.get(label, 0))
                else:
                    aligned_labels.append(0)  # 'O' 标签
            else:
                # 同一个词的后续subword
                if word_idx < len(labels):
                    label = labels[word_idx]
                    # B-标签改为I-标签，I-标签保持不变，O标签保持不变
                    if label.startswith('B-'):
                        i_label = 'I-' + label[2:]
                        aligned_labels.append(self.label_to_index.get(i_label, 0))
                    else:
                        aligned_labels.append(self.label_to_index.get(label, 0))
                else:
                    aligned_labels.append(0)
                    
            previous_word_idx = word_idx
        
        # 转换为tensor
        label_ids = torch.LongTensor(aligned_labels)
        
        return input_ids, label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    """加载词汇表"""
    if not os.path.exists(vocab_path):
        return {}
    
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置
    return token_dict


def load_data(data_path, config, shuffle=True):
    """加载NER数据"""
    dg = NERDataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = NERDataGenerator(Config["train_data_path"], Config)
    print(f"数据量: {len(dg)}")
    print(f"类别数: {dg.config['class_num']}")
    print(f"标签映射: {dg.label_to_index}")
    if len(dg) > 0:
        input_ids, label_ids = dg[0]
        print(f"输入样本形状: {input_ids.shape}")
        print(f"标签样本形状: {label_ids.shape}")
        print(f"输入示例: {input_ids[:10]}")
        print(f"标签示例: {label_ids[:10]}")
