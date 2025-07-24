# loader.py
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config import global_config as config
import torch


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 使用BERT tokenizer编码文本
        encoding = config.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def clean_text(text):
    """基础文本清洗"""
    if not isinstance(text, str):
        return ""
    # 移除特殊字符，保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5\w\s，。！？；："\'、]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_process_data():
    """加载并处理数据"""
    # 加载数据
    try:
        df = pd.read_csv(config.data_path)
    except Exception as e:
        raise FileNotFoundError(f"加载数据失败: {e}")

    # 检查列名
    if config.text_column not in df.columns or config.label_column not in df.columns:
        # 尝试自动识别列
        text_col = df.columns[0]
        label_col = df.columns[1]
        df = df.rename(columns={
            text_col: config.text_column,
            label_col: config.label_column
        })

    # 确保标签是整数 (0和1)
    df[config.label_column] = df[config.label_column].astype(int)

    # 数据清洗
    df['cleaned_text'] = df[config.text_column].apply(clean_text)

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[config.label_column])

    # 创建数据集
    train_dataset = TextDataset(
        texts=train_df['cleaned_text'].values,
        labels=train_df[config.label_column].values,
        tokenizer=config.tokenizer,
        max_len=config.max_sequence_length
    )

    val_dataset = TextDataset(
        texts=val_df['cleaned_text'].values,
        labels=val_df[config.label_column].values,
        tokenizer=config.tokenizer,
        max_len=config.max_sequence_length
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # 在Windows上可能需要设为0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def analyze_data():
    """数据分析"""
    try:
        df = pd.read_csv(config.data_path)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return {}

    # 检查列名
    if config.text_column not in df.columns or config.label_column not in df.columns:
        text_col = df.columns[0]
        label_col = df.columns[1]
        df = df.rename(columns={
            text_col: config.text_column,
            label_col: config.label_column
        })

    # 正负样本分析
    positive_count = (df[config.label_column] == 1).sum()
    negative_count = (df[config.label_column] == 0).sum()

    # 文本长度分析
    df['cleaned_text'] = df[config.text_column].apply(clean_text)
    text_lengths = df['cleaned_text'].apply(len)  # 字符长度

    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'avg_length': text_lengths.mean(),
        'max_length': text_lengths.max(),
        'min_length': text_lengths.min()
    }