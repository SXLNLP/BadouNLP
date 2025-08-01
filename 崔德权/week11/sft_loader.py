# -*- coding: utf-8 -*-

import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any

class NewsDataset(Dataset):
    """新闻数据集类，用于SFT训练"""
    
    def __init__(self, data_path: str, tokenizer, config: Dict[str, Any], split_ratio: float = 0.9):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config["max_length"]
        self.prompt_template = config["prompt_template"]
        
        # 加载数据
        self.data = self.load_data(data_path, split_ratio)
        
    def load_data(self, data_path: str, split_ratio: float) -> List[Dict[str, str]]:
        """加载新闻数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'title' in item and 'content' in item:
                        data.append({
                            'title': item['title'].strip(),
                            'content': item['content'].strip()
                        })
                except json.JSONDecodeError:
                    continue
        
        # 数据分割（简单随机分割）
        random.shuffle(data)
        split_idx = int(len(data) * split_ratio)
        return data[:split_idx]  # 返回训练部分
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']
        title = item['title']
        
        # 构建输入文本
        input_text = self.prompt_template.format(content=content)
        target_text = title
        
        # 编码输入和目标
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.config["output_max_length"],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0),
            'target_attention_mask': targets['attention_mask'].squeeze(0)
        }

class NewsValidationDataset(Dataset):
    """新闻验证数据集类"""
    
    def __init__(self, data_path: str, tokenizer, config: Dict[str, Any], split_ratio: float = 0.9):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config["max_length"]
        self.prompt_template = config["prompt_template"]
        
        # 加载数据
        self.data = self.load_data(data_path, split_ratio)
        
    def load_data(self, data_path: str, split_ratio: float) -> List[Dict[str, str]]:
        """加载新闻数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'title' in item and 'content' in item:
                        data.append({
                            'title': item['title'].strip(),
                            'content': item['content'].strip()
                        })
                except json.JSONDecodeError:
                    continue
        
        # 数据分割（简单随机分割）
        random.shuffle(data)
        split_idx = int(len(data) * split_ratio)
        return data[split_idx:]  # 返回验证部分
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        content = item['content']
        title = item['title']
        
        # 构建输入文本
        input_text = self.prompt_template.format(content=content)
        target_text = title
        
        # 编码输入和目标
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.config["output_max_length"],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0),
            'target_attention_mask': targets['attention_mask'].squeeze(0)
        }

def create_dataloaders(config: Dict[str, Any]):
    """创建训练和验证数据加载器"""
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model_path"])
    
    # 添加特殊token
    special_tokens = {
        'pad_token': '[PAD]',
        'unk_token': '[UNK]',
        'bos_token': '[CLS]',
        'eos_token': '[SEP]'
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 创建数据集
    train_dataset = NewsDataset(config["train_data_path"], tokenizer, config)
    val_dataset = NewsValidationDataset(config["valid_data_path"], tokenizer, config)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 