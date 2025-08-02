#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertConfig, BertModel
from transformers import BertTokenizer, BertForMaskedLM

"""
基于BERT+Mask的自回归语言模型训练
"""

class LanguageModel(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModel, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.config.is_decoder = True  # 启用自回归模式
        self.bert = BertModel.from_pretrained(bert_path, config=self.config)
        self.cls = BertForMaskedLM(self.config).cls
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充位置

    def forward(self, x, labels=None):
        attention_mask = (x != 0).float()  # 假设0是填充标记
        
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True  # 确保返回字典
        )
        
        # 获取隐藏状态
        sequence_output = outputs[0]  # 元组中的第一个元素是隐藏状态
        
        # 通过分类头获取logits
        logits = self.cls(sequence_output)
        
        if labels is not None:
            # 计算损失 - 注意：labels的维度应与logits匹配
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 展平logits和labels
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_labels.view(-1))
            return loss
        else:
            return logits

# 加载字表
def build_vocab(vocab_path):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    with open(vocab_path, encoding="utf8") as f:
        for line in f:
            char = line.strip()
            # 跳过特殊标记（如果已经存在）
            if char not in vocab:
                vocab[char] = len(vocab)
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
    
    # 输入序列：添加[CLS]和[SEP]
    input_seq = "[CLS]" + window
    x = [vocab.get(char, vocab["[UNK]"]) for char in input_seq]
    
    # 标签序列：右移一位
    target_seq = window
    y = [vocab.get(char, vocab["[UNK]"]) for char in target_seq]
    
    # 确保长度一致
    if len(x) > len(y):
        y.extend([-100] * (len(x) - len(y)))
    elif len(y) > len(x):
        y = y[:len(x)]
    
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 文本生成测试
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = {idx: char for char, idx in vocab.items()}
    model.eval()
    
    # 初始输入
    input_str = "[CLS]" + openings
    generated = openings
    
    with torch.no_grad():
        while len(generated) <= 50:  # 限制生成长度
            # 准备输入
            input_ids = [vocab.get(char, vocab["[UNK]"]) 
                        for char in input_str[-window_size:]]  # 截取窗口大小
            
            # 填充到窗口大小
            if len(input_ids) < window_size:
                input_ids = [vocab["[PAD]"]] * (window_size - len(input_ids)) + input_ids
            
            input_tensor = torch.LongTensor([input_ids])
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # 模型预测
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            
            # 采样策略
            if random.random() > 0.1:
                next_token = torch.argmax(next_token_logits).item()
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            # 转换token并更新
            next_char = reverse_vocab.get(next_token, "[UNK]")
            if next_char in ["[SEP]", "\n"] or len(generated) >= 50:
                break
                
            generated += next_char
            input_str += next_char
    
    return generated

def train(corpus_path, bert_path, save_weight=True):
    epoch_num = 5         # 训练轮数
    batch_size = 8        # 批次大小（减小以适应M1）
    train_sample = 2000   # 每轮训练样本数（减小以适应M1）
    window_size = 16      # 文本长度（减小以适应M1）
    
    vocab_path = os.path.join(bert_path, "vocab.txt")
    vocab = build_vocab(vocab_path)
    corpus = load_corpus(corpus_path)
    model = LanguageModel(bert_path)
    
    # 苹果M1芯片使用MPS加速
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print("使用Apple M1 GPU (MPS)训练")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
        print("使用NVIDIA GPU训练")
    else:
        device = torch.device("cpu")
        print("使用CPU训练")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    print("模型加载完毕，开始训练...")
    print(f"词表大小: {len(vocab)}")
    print(f"语料长度: {len(corpus)}")
    
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        batch_count = int(train_sample / batch_size)
        
        for batch in range(batch_count):
            # 构建训练数据
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            loss = model(x, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch % 50 == 0:
                print(f"Epoch {epoch+1} Batch {batch}/{batch_count} Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
        
        # 生成示例
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    
    if save_weight:
        torch.save(model.state_dict(), "bert_lm.pth")
        print("模型已保存到 bert_lm.pth")

if __name__ == "__main__":
    bert_path = r'/Users/chenayu/Desktop/111/bert-base-chinese'
    train("corpus.txt", bert_path, save_weight=True)
