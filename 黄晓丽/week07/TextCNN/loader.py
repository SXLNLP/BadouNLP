# loader.py
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter
import torch
from config import global_config as config  # 使用全局配置


class TextDataset:
    def __init__(self):
        self.vocab = None
        self.vocab_size = None

    def clean_text(self, text):
        """清洗文本"""
        if not isinstance(text, str):
            return ""
            # 移除所有非中文字符、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5\w\s，。！？；："\'、]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chinese_tokenizer(self, text):
        """中文分词"""
        tokens = list(jieba.cut(text))
        # 过滤空字符串
        return [token for token in tokens if token.strip()]

    def build_vocab(self, tokens_list):
        """构建词汇表（增强安全性）"""
        word_counts = Counter()
        for tokens in tokens_list:
            word_counts.update(tokens)

        # 确保特殊标记存在
        special_tokens = ['<PAD>', '<UNK>']
        vocab = {}

        # 添加特殊标记
        for i, token in enumerate(special_tokens):
            vocab[token] = i

        # 添加常规词汇
        idx = len(special_tokens)
        for word, count in word_counts.items():
            if count >= config.min_word_freq:
                vocab[word] = idx
                idx += 1

        return vocab

    def text_to_sequence(self, tokens, max_len):
        """文本转索引序列（添加索引范围检查）"""
        sequence = []
        for token in tokens:
            # 确保索引在词汇表范围内
            idx = self.vocab.get(token, self.vocab['<UNK>'])
            if idx >= len(self.vocab):
                idx = self.vocab['<UNK>']  # 强制设为未知词
            sequence.append(idx)

        # 填充或截断
        if len(sequence) < max_len:
            sequence = sequence + [self.vocab['<PAD>']] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        return sequence

    def load_and_process_data(self):
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

        # 数据清洗和分词
        df['cleaned_text'] = df[config.text_column].apply(self.clean_text)
        df['tokens'] = df['cleaned_text'].apply(self.chinese_tokenizer)

        # 构建词汇表
        self.vocab = self.build_vocab(df['tokens'])
        self.vocab_size = len(self.vocab)

        # 更新全局配置
        config.vocab_size = self.vocab_size

        # 文本转序列
        df['sequence'] = df['tokens'].apply(
            lambda x: self.text_to_sequence(x, config.max_sequence_length))

        # 验证索引范围 (添加此部分)
        all_indices = [idx for seq in df['sequence'] for idx in seq]
        max_index = max(all_indices)
        min_index = min(all_indices)
        print(f"索引范围: {min_index} - {max_index}, 词汇表大小: {len(self.vocab)}")

        if max_index >= len(self.vocab):
            print(f"警告: 发现超出词汇表范围的索引 {max_index} (词汇表大小 {len(self.vocab)})")
            # 修复超出范围的索引
            df['sequence'] = df['sequence'].apply(
                lambda seq: [idx if idx < len(self.vocab) else self.vocab['<UNK>'] for idx in seq])

        # 划分训练集和验证集
        X = np.array(df['sequence'].tolist())
        y = np.array(df[config.label_column].astype(int).tolist())  # 确保标签是整数

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # 转换为PyTorch张量
        X_train_tensor = torch.LongTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.LongTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        return train_loader, val_loader, self.vocab

    def analyze_data(self):
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
        df['cleaned_text'] = df[config.text_column].apply(self.clean_text)
        df['tokens'] = df['cleaned_text'].apply(self.chinese_tokenizer)
        text_lengths = df['tokens'].apply(len)

        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'avg_length': text_lengths.mean(),
            'max_length': text_lengths.max(),
            'min_length': text_lengths.min()
        }