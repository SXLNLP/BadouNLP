# lstm_classification.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np

# 文本处理器
class TextProcessor:
    def __init__(self, max_len=50, max_vocab=10000):
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        
    def _tokenize(self, text):
        """使用jieba分词"""
        return list(jieba.cut(text.strip()))
    
    def build_vocab(self, texts):
        """构建词汇表"""
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)
            
        # 添加高频词
        for word, freq in counter.most_common(self.max_vocab):
            if word not in self.vocab and len(self.vocab) < self.max_vocab:
                self.vocab[word] = len(self.vocab)
                
    def tokenize_and_encode(self, text):
        """文本转ID序列"""
        tokens = self._tokenize(text)
        encoded = [self.vocab.get(t, 1) for t in tokens[:self.max_len]]
        return encoded + [0]*(self.max_len - len(encoded))

# 自定义数据集类
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = texts
        self.labels = labels
        self.processor = processor
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.processor.tokenize_and_encode(text)
        return torch.tensor(encoded), torch.tensor(label)

# LSTM模型定义
class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim*2, num_classes)
        )
        
    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        x = x.mean(dim=1)    # [batch_size, hidden_dim*2]
        return self.classifier(x)

# 数据加载函数
def load_data(file_path):
    df = pd.read_csv(file_path, header=0, names=['label', 'text'])
    texts = df['text'].tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

# 模型训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # 计算准确率
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return total_loss/len(dataloader), correct/len(dataloader)

if __name__ == "__main__":
    # 超参数
    MAX_LEN = 50
    BATCH_SIZE = 32
    EMBED_DIM = 256
    HIDDEN_DIM = 128
    LR = 1e-3
    EPOCHS = 10
    
    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    texts, labels = load_data('F:/BaiduNetdiskDownload/八斗精品班/第七周 文本分类/week7 文本分类问题/文本分类练习.csv')
    
    # 构建处理器
    processor = TextProcessor(max_len=MAX_LEN)
    processor.build_vocab(texts)
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # 创建数据集
    train_dataset = ReviewDataset(X_train, y_train, processor)
    val_dataset = ReviewDataset(X_val, y_val, processor)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    vocab_size = len(processor.vocab)
    model = LSTMTextClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
    print(f"使用GPU: {next(model.parameters()).is_cuda}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 训练循环
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
    
    # 模型评估
    print("\n=== 模型评估 ===")
    preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    
    print(classification_report(y_val, preds))
    
    # 保存模型
    torch.save(model.state_dict(), "lstm_model.pth")
    print("模型已保存！")
