# -*- coding: utf-8 -*-
import json  
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast  
class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用BERT快速分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                # 跳过空段落
                if segment.strip() == "":
                    continue
                    
                sentence = []
                labels = []
                lines = segment.split("\n")
                
                # 确保每行数据格式正确
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        # 跳过格式错误的数据行
                        continue
                    char, label = parts[0], parts[1]
                    sentence.append(char)
                    labels.append(self.schema[label])
                
                if not sentence:  # 确保有实际数据
                    continue
                    
                self.sentences.append("".join(sentence))
                
                # 使用BERT tokenizer编码
                encoding = self.tokenizer(
                    sentence,
                    padding='max_length',
                    max_length=self.config["max_length"],
                    truncation=True,
                    is_split_into_words=True,
                    return_tensors='pt'
                )
                
                # 获取word_ids用于对齐标签
                word_ids = encoding.word_ids()  # 直接调用word_ids()
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    # 特殊token设为-100
                    if word_idx is None:
                        label_ids.append(-100)
                    # 每个词的第一个token分配标签
                    elif word_idx != previous_word_idx:
                        label_ids.append(labels[word_idx])
                    # 同一个词的其他token分配-100
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                # 确保标签长度与输入一致
                while len(label_ids) < self.config["max_length"]:
                    label_ids.append(-100)
                
                # 确保第一个位置有效（解决CRF问题）
                if encoding["attention_mask"][0][0].item() == 0:
                    # 如果第一个位置是填充符(PAD)，强制设为有效位置
                    encoding["attention_mask"][0][0] = 1
                    # 将对应的标签设为特殊标签（如O标签）
                    label_ids[0] = self.schema.get("O", 0)
                
                self.data.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "token_type_ids": encoding["token_type_ids"].squeeze(0),
                    "labels": torch.LongTensor(label_ids)
                })
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        # 从JSON文件加载标签schema
        with open(path, encoding="utf8") as f:
            return json.load(f) 
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
