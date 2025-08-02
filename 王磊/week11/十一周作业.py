import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # 创建位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置序列
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)  # 频率计算
        pe[:, 0::2] = torch.sin(position * div_term)  # 正弦编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 余弦编码
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)  # 注册为缓冲区

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # 添加位置编码到输入


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab, nhead=8, num_layers=6):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_dim)  # 词嵌入层
        self.pos_encoder = PositionalEncoding(input_dim)  # 位置编码
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, dim_feedforward=input_dim * 4)  # Transformer层
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)  # Transformer编码器
        self.classify = nn.Linear(input_dim, len(vocab))  # 分类层
        self.dropout = nn.Dropout(0.1)  # Dropout层
        self.loss = nn.functional.cross_entropy  # 损失函数
        self.input_dim = input_dim
        self.mask = self.generate_square_subsequent_mask(100)  # 生成因果mask（上三角矩阵）

    def generate_square_subsequent_mask(self, sz):
        """生成因果mask（仅允许关注左侧token）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1) == 1  # 上三角矩阵（含对角线）
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # True的位置设为负无穷
        return mask

    def forward(self, x, y=None):
        seq_len = x.size(1)
        device = x.device
        # 动态调整mask大小以适应输入序列
        mask = self.mask[:seq_len, :seq_len].to(device)  # 截取所需大小的mask

        x = self.embedding(x) * math.sqrt(self.input_dim)  # 词嵌入
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.dropout(x)  # 应用dropout
        x = x.transpose(0, 1)  # 调整维度为(seq_len, batch, dim)
        x = self.transformer_encoder(x, mask)  # Transformer处理（应用因果mask）
        x = x.transpose(0, 1)  # 恢复维度为(batch, seq_len, dim)
        y_pred = self.classify(x)  # 预测输出

        if y is not None:
            # SFT训练：只计算response部分的损失
            # 假设输入格式为[instruction, response]，通过特殊标记分隔
            sep_token_id = vocab.get('\t', 1)  # 获取分隔符ID
            # 创建损失掩码（response部分为1，instruction部分为0）
            mask = (x == sep_token_id).int().argmax(dim=1)  # 找到分隔符位置
            loss_mask = torch.arange(seq_len).expand_as(y) > mask.unsqueeze(1)  # 生成掩码

            # 计算掩码位置的平均损失
            loss = self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), reduction='none')
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()  # 只计算response部分
            return loss
        else:
            return torch.softmax(y_pred, dim=-1)  # 推理时返回概率分布


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<UNK>": 1}  # 初始化词表（添加UNK标记）
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()  # 处理换行符
            vocab[char] = index + 2  # 从2开始编号（0和1已占用）
    # 添加SFT所需特殊标记
    vocab['\t'] = len(vocab)  # 指令-回复分隔符
    vocab['<EOS>'] = len(vocab)  # 结束标记
    return vocab


# 加载SFT语料（指令-回复对）
def load_sft_corpus(path):
    samples = []
    with open(path, encoding="gbk") as f:
        for line in f:
            parts = line.strip().split('\t')  # 按制表符分割指令和回复
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))
    return samples


# 构建SFT样本（格式：instruction + \t + response + <EOS>）
def build_sft_sample(vocab, max_len, corpus):
    inst, resp = random.choice(corpus)  # 随机选择样本
    # 拼接完整序列
    sequence = inst + '\t' + resp + '<EOS>'
    # 转换为ID序列
    x = [vocab.get(c, vocab["<UNK>"]) for c in sequence[:max_len]]
    y = x[1:] + [vocab["<pad>"]]  # 目标序列（右移一位）
    return x, y[:-1]  # 保证x和y等长


# 建立SFT数据集
def build_sft_dataset(sample_count, vocab, max_len, corpus):
    dataset_x, dataset_y = [], []
    for _ in range(sample_count):
        x, y = build_sft_sample(vocab, max_len, corpus)
        # 填充到固定长度
        x = x + [vocab["<pad>"]] * (max_len - len(x))
        y = y + [vocab["<pad>"]] * (max_len - len(y))
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 构建模型
def build_model(vocab, char_dim):
    return LanguageModel(char_dim, vocab)


# 文本生成（适配SFT格式）
def generate_sentence(instruction, model, vocab, max_len):
    reverse_vocab = {idx: char for char, idx in vocab.items()}
    model.eval()
    with torch.no_grad():
        # 初始化输入（指令 + 分隔符）
        input_seq = [vocab.get(c, vocab["<UNK>"]) for c in instruction + '\t']
        while len(input_seq) < max_len:
            x = torch.LongTensor([input_seq]).to(next(model.parameters()).device)
            y_pred = model(x)[0][-1]  # 获取最后一个token预测
            next_token = sampling_strategy(y_pred)  # 采样策略
            char = reverse_vocab.get(next_token, "<UNK>")
            if char == '<EOS>':  # 遇到结束符停止
                break
            input_seq.append(next_token)
        # 提取回复部分（分隔符之后的内容）
        response = ''.join([reverse_vocab[idx] for idx in input_seq])
        return response.split('\t')[-1].replace('<EOS>', '')


# 采样策略
def sampling_strategy(probs):
    if random.random() > 0.1:  # 90%贪心搜索
        return torch.argmax(probs).item()
    else:  # 10%随机采样
        probs = torch.softmax(probs, dim=-1).cpu().numpy()
        return np.random.choice(len(probs), p=probs)


# 训练函数
def train_sft(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 批次大小
    samples_per_epoch = 50000  # 每轮样本数
    char_dim = 256  # 字符维度
    max_len = 50  # 最大序列长度

    vocab = build_vocab("vocab.txt")  # 构建词表
    sft_corpus = load_sft_corpus(corpus_path)  # 加载SFT语料
    model = build_model(vocab, char_dim)  # 构建模型

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器

    print("SFT训练启动...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = []
        for _ in range(samples_per_epoch // batch_size):
            # 构建批次数据
            x, y = build_sft_dataset(batch_size, vocab, max_len, sft_corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            loss = model(x, y)  # 前向计算（含损失）
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            total_loss.append(loss.item())

        # 打印训练进度
        avg_loss = np.mean(total_loss)
        print(f"Epoch {epoch + 1}/{epoch_num} | Loss: {avg_loss:.4f}")

        # 示例生成
        print(generate_sentence("请写一首关于春天的诗：", model, vocab, max_len))
        print(generate_sentence("解释量子计算：", model, vocab, max_len))

    # 保存模型
    if save_weight:
        torch.save(model.state_dict(), "sft_model.pth")


if __name__ == "__main__":
    train_sft("sft_corpus.txt")  # 使用SFT格式语料
