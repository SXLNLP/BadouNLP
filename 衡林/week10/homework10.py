#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel, BertConfig

"""
基于pytorch的BERT语言模型（使用掩码实现自回归）
"""

def get_subsequent_mask(seq):
    '''生成下三角掩码矩阵'''
    len_s = seq.size(1)
    subsequent_mask = torch.tril(torch.ones((len_s, len_s), device=seq.device)).bool()
    return subsequent_mask.unsqueeze(0)  # 增加batch维度

class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        # 使用本地路径加载BERT模型
        self.bert = BertModel.from_pretrained(r'E:\八斗精品班\第六周 语言模型\bert-base-chinese\bert-base-chinese')
        self.classify = nn.Linear(self.bert.config.hidden_size, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # 生成自回归掩码
        amsk = get_subsequent_mask(x)
        
        # BERT前向传播
        outputs = self.bert(
            input_ids=x,
            attention_mask=amsk.squeeze(0)  # 适配BERT的掩码形状要求
        
        # 获取最后一层隐藏状态
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 分类层
        y_pred = self.classify(sequence_output)
        
        if y is not None:
            # 计算损失
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    window = corpus[start:start+window_size]
    target = corpus[start+1:start+window_size+1]  # 输入输出错开一位
    
    # 转换为ID序列
    x = [vocab.get(char, vocab.get("<UNK>", 1)) for char in window]
    y = [vocab.get(char, vocab.get("<UNK>", 1)) for char in target]
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x, dataset_y = [], []
    for _ in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab):
    return LanguageModel(vocab)

# 文本生成函数
def generate_sentence(openings, model, vocab, window_size, device):
    reverse_vocab = {idx: char for char, idx in vocab.items()}
    model.eval()
    generated = openings
    
    with torch.no_grad():
        while len(generated) <= 50:  # 限制最大生成长度
            # 准备输入
            input_chars = generated[-window_size:] if len(generated) >= window_size else generated
            x = [vocab.get(char, vocab.get("<UNK>", 1)) for char in input_chars]
            x = torch.LongTensor([x]).to(device)
            
            # 预测下一个字符
            y_pred = model(x)[0, -1]
            next_char_idx = sampling_strategy(y_pred)
            next_char = reverse_vocab.get(next_char_idx, "<UNK>")
            
            # 终止条件
            if next_char == "\n" or len(generated) > 50:
                break
                
            generated += next_char
            
    return generated

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        return torch.argmax(prob_distribution).item()
    else:
        probs = torch.softmax(prob_distribution, dim=-1).cpu().numpy()
        return np.random.choice(len(probs), p=probs)

# 训练函数
def train(corpus_path, save_weight=True):
    # 训练参数
    epoch_num = 20
    batch_size = 32  # 适当减小batch_size以防内存不足
    train_sample = 50000
    window_size = 100
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据准备
    vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    
    # 模型初始化
    model = build_model(vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    print("模型加载完毕，开始训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = []
        
        for _ in range(int(train_sample / batch_size)):
            # 构建训练批次
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            x, y = x.to(device), y.to(device)
            
            # 训练步骤
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        
        # 打印训练信息
        avg_loss = np.mean(total_loss)
        print(f"Epoch {epoch+1}/{epoch_num}, Loss: {avg_loss:.4f}")
        
        # 生成示例文本
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size, device))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size, device))
    
    # 保存模型
    if save_weight:
        torch.save(model.state_dict(), "bert_lm.pth")

if __name__ == "__main__":
    train("corpus.txt", False)
