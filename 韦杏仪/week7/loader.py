import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from config import config


# 文本分词
def tokenize_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return text.split()


# 构建词表
def build_vocab(texts, vocab_size=config.vocab_size):
    all_words = []
    for text in texts:
        all_words.extend(tokenize_text(text))

    word_counts = Counter(all_words)
    vocab = {'<pad>': 0, '<unk>': 1}

    for word, _ in word_counts.most_common(vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab


# 将文本转换为词表中的索引
def text_to_indices(text, vocab, max_len=config.max_seq_length):
    tokens = tokenize_text(text)
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]

    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))  # 用<pad>（0）填充
    else:
        indices = indices[:max_len]  # 截断过长的文本
    return indices


# 数据分析函数
def analyze_data(texts, labels):
    # 统计正负样本数量,标签1是好评，0是差评
    label_counts = Counter(labels)
    print(f"数据集中共有 {len(texts)} 条评论")
    print(f"好评({1}): {label_counts[1]}, 差评({0}): {label_counts[0]}")

    # 计算文本长度
    text_lengths = [len(tokenize_text(text)) for text in texts]
    print(f"平均文本长度: {np.mean(text_lengths):.2f} 个词")
    print(f"最长文本: {max(text_lengths)} 个词，最短文本: {min(text_lengths)} 个词")


# 自定义数据集类
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 将文本转换为索引
        indices = text_to_indices(text, self.vocab)
        # 转换为PyTorch张量
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 加载数据并划分训练集和验证集
def load_data():
    print("正在加载数据...")
    # 读取CSV文件
    df = pd.read_csv(config.data_path)

    print(f"数据基本信息：")
    df.info()

    # 数据探查：查看数据集行数和列数
    rows, columns = df.shape

    label_column = None
    if '标签' in df.columns:
        label_column = '标签'
    elif columns >= 2:
        # 假设第二列是标签
        print("警告：未找到'标签'列，将使用第二列作为标签")
        label_column = df.columns[1]
    else:
        raise ValueError("无法确定标签列，请检查数据集结构")

    # 数据探查：查看标签列的基本统计信息
    print(f"\n标签列 '{label_column}' 的基本统计：")
    print(df[label_column].value_counts())

    # 尝试转换标签为整数
    try:
        labels = df[label_column].astype(int).tolist()
    except ValueError as e:
        print(f"错误：标签列无法直接转换为整数。错误信息：{e}")
        print("尝试手动映射标签...")

        # 获取标签列的唯一值
        unique_labels = df[label_column].unique()
        print(f"发现以下标签值：{unique_labels}")

        # 简单映射
        label_map = {label: 1 if '好' in str(label) else 0 for label in unique_labels}
        print(f"使用标签映射：{label_map}")

        # 应用映射
        labels = df[label_column].map(label_map).tolist()

        # 检查是否所有标签都成功映射
        if any(pd.isna(labels)):
            raise ValueError("无法将所有标签映射为0或1，请检查数据集")

    # 获取文本列
    if '评论' in df.columns:
        texts = df['评论'].astype(str).tolist()
    else:
        # 假设第一列是文本
        print("警告：未找到'评论'列，将使用第一列作为文本")
        texts = df.iloc[:, 0].astype(str).tolist()

    # 数据分析
    analyze_data(texts, labels)

    # 划分训练集和验证集
    n = len(texts)
    n_val = int(config.test_size * n)

    indices = list(range(n))
    np.random.seed(config.seed)
    np.random.shuffle(indices)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = [texts[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_val = [texts[i] for i in val_indices]
    y_val = [labels[i] for i in val_indices]

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    return X_train, X_val, y_train, y_val


# 创建数据加载器
def create_data_loaders(X_train, X_val, y_train, y_val):
    print("正在创建数据加载器...")
    # 构建词表
    vocab = build_vocab(X_train)
    print(f"词表大小: {len(vocab)} 个词")

    # 创建数据集
    train_dataset = ReviewDataset(X_train, y_train, vocab)
    val_dataset = ReviewDataset(X_val, y_val, vocab)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, vocab
