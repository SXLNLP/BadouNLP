# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/5/28
# @Author      : liuboyuan
# @Description : 使用RNN进行字符串中字符'a'位置分类任务

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from classifier_ import TorchModel

class StringDataset(Dataset):
    def __init__(self, strings, labels, vocab):
        self.strings = strings
        self.labels = labels
        self.vocab = vocab
        
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        string = self.strings[idx]
        label = self.labels[idx]
        
        # 将字符串转换为索引
        indices = [self.vocab.get(char, self.vocab['<UNK>']) for char in string]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_vocab(strings):
    """构建词汇表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for string in strings:
        for char in string:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

def generate_sample_data(max_length=10):
    """生成单个样本：包含字符'a'的随机字符串"""
    # 随机选择字符串长度（至少为1）
    length = random.randint(1, max_length)
    
    # 随机选择'a'第一次出现的位置
    a_position = random.randint(0, length - 1)
    
    # 生成字符串
    chars = []
    for i in range(length):
        if i == a_position:
            chars.append('a')
        elif i > a_position:
            # 'a'出现后，可以随机选择任意字符（包括'a'）
            chars.append(random.choice(string.ascii_lowercase))
        else:
            # 'a'出现前，不能包含'a'
            chars.append(random.choice([c for c in string.ascii_lowercase if c != 'a']))
    
    return ''.join(chars), a_position

def build_dataset(sample_count, max_length=10):
    """构建数据集"""
    strings = []
    labels = []
    
    for _ in range(sample_count):
        string, label = generate_sample_data(max_length)
        strings.append(string)
        labels.append(label)
    
    return strings, labels

def pad_sequences(sequences, max_length, pad_value=0):
    """填充序列到统一长度"""
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            padded.append(seq + [pad_value] * (max_length - len(seq)))
        else:
            padded.append(seq[:max_length])
    return torch.tensor(padded, dtype=torch.long)

def collate_fn(batch):
    """自定义批处理函数"""
    sequences, labels = zip(*batch)
    
    # 找到批次中的最大长度
    max_length = max(len(seq) for seq in sequences)
    
    # 填充序列
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = torch.cat([seq, torch.zeros(max_length - len(seq), dtype=torch.long)])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    return torch.stack(padded_sequences), torch.stack(labels)

def train_model(model, train_loader, optimizer, device):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        loss = model(data, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 获取预测结果
        with torch.no_grad():
            predictions = model(data)
            predicted_classes = torch.argmax(predictions, dim=1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            predictions = model(data)
            predicted_classes = torch.argmax(predictions, dim=1)
            
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

def plot_metrics(train_losses, train_accuracies):
    """绘制训练指标"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

def predict_samples(model, test_strings, vocab, device, max_length=10):
    """对测试样本进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for string in test_strings:
            # 转换为索引
            indices = [vocab.get(char, vocab['<UNK>']) for char in string]
            
            # 填充或截断到指定长度
            if len(indices) < max_length:
                indices.extend([vocab['<PAD>']] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
            
            # 转换为tensor并添加batch维度
            input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
            
            # 预测
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            predictions.append(predicted_class)
    
    return predictions

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数设置
    max_length = 10
    train_sample_count = 5000
    test_sample_count = 1000
    batch_size = 32
    epochs = 20
    learning_rate = 0.001
    vector_dim = 64
    hidden_dim = 128
    
    print("📊 开始构建数据集...")
    
    # 构建训练数据集
    train_strings, train_labels = build_dataset(train_sample_count, max_length)
    
    # 构建测试数据集
    test_strings, test_labels = build_dataset(test_sample_count, max_length)
    
    # 构建词汇表
    all_strings = train_strings + test_strings
    vocab = build_vocab(all_strings)
    print(f"词汇表大小: {len(vocab)}")
    print(f"词汇表: {vocab}")
    
    # 计算类别数（最大位置索引 + 1）
    num_classes = max_length
    print(f"分类类别数: {num_classes}")
    
    # 创建数据集和数据加载器
    train_dataset = StringDataset(train_strings, train_labels, vocab)
    test_dataset = StringDataset(test_strings, test_labels, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    model = TorchModel(vector_dim, num_classes, vocab, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("🚀 开始训练...")
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # 评估模型
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"🎯 测试集准确率: {test_accuracy:.4f}")
    
    # 绘制训练曲线
    plot_metrics(train_losses, train_accuracies)
    
    # 保存模型
    torch.save(model.state_dict(), 'rnn_classifier_model.pth')
    print("💾 模型已保存为 rnn_classifier_model.pth")
    
    # 测试预测功能
    print("\n🔮 对测试样本进行预测:")
    test_samples = [
        "abcdef",     # 'a'在位置0
        "bcadef",     # 'a'在位置2
        "bcdaef",     # 'a'在位置3
        "bcdefga",    # 'a'在位置6
        "a",          # 只有'a'，在位置0
    ]
    
    predictions = predict_samples(model, test_samples, vocab, device, max_length)
    
    for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
        actual_pos = sample.index('a') if 'a' in sample else -1
        print(f"样本 {i+1}: '{sample}' → 预测位置: {pred}, 实际位置: {actual_pos}")

if __name__ == "__main__":
    main() 