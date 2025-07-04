import json
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from collections import defaultdict

'''
数据加载，CSV文件处理
'''


def csv_to_json(csv_file_path, test_size):
    '''
    处理CSV文件，转换为JSON格式并划分训练集和验证集
    '''
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    # print(df.columns.tolist())
    # print(df.shape)

    # 划分训练集和验证集
    train_df, valid_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])

    # 保存为JSON格式
    def save_to_json(data, filename):
        with open(filename, 'w', encoding='utf8') as f:
            for _, row in data.iterrows():
                tag = '好评' if row['label'] == 1 else '差评'
                item = {
                    'tag': tag,
                    'title': str(row['review'])
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_to_json(train_df, './data/train_data_news.json')
    save_to_json(valid_df, './data/valid_data_news.json')
    # print(train_df)
    # print(len(train_df))
    # print(len(valid_df))
    return len(train_df), len(valid_df)


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = {'差评': 0, '好评': 1}
        self.config['class_num'] = len(self.index_to_label)

        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        else:
            self.vocab = load_vocab(config['vocab_path'])
            self.config['vocab_size'] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                tag = line['tag']
                label = self.label_to_index[tag]
                title = line['title']
                if self.config['model_type'] == 'bert':
                    input_id = self.tokenizer.encode(title, max_length=self.config['max_length'],
                                                     padding='max_length', truncation=True)
                else:
                    input_id = self.sentence_to_sequence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def sentence_to_sequence(self, title):
        input_id = []
        for char in title:
            input_id.append(self.vocab.get(char, self.vocab['[unk]']))
        input_id = self.padding(input_id)
        return input_id

    # 保证形状一致
    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    """加载词汇表并确保包含必要的特殊标记"""
    token_dict = {}
    
    # 首先添加特殊标记
    token_dict['[pad]'] = 0
    token_dict['[unk]'] = 1
    current_idx = 2  # 从2开始，因为0和1已经被占用
    
    try:
        with open(vocab_path, encoding='utf8') as file:
            for line in file:
                token = line.strip()
                # 跳过已经添加的特殊标记
                if token in token_dict:
                    continue
                token_dict[token] = current_idx
                current_idx += 1
    except FileNotFoundError:
        raise FileNotFoundError(f"词汇表文件未找到: {vocab_path}")
    
    # 验证特殊标记是否存在
    required_special_tokens = ['[pad]', '[unk]']
    for token in required_special_tokens:
        if token not in token_dict:
            raise ValueError(f"词汇表缺少必要的特殊标记: {token}")
    
    print(f"词汇表加载完成，大小: {len(token_dict)}")
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    data_gen = DataGenerator(data_path, config)
    data_loader = DataLoader(data_gen, batch_size=config['batch_size'], shuffle=shuffle)
    return data_loader


if __name__ == '__main__':
    # 处理CSV文件
    csv_file = './data/文本分类练习.csv'
    csv_to_json(csv_file, 0.5)
