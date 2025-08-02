# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import re
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""
基于Transformer的自回归语言模型（类似GPT2）
"""


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab, nhead=8, num_layers=6, max_len=500):
        super(LanguageModel, self).__init__()
        self.vocab_size = len(vocab)
        self.input_dim = input_dim
        self.max_len = max_len

        # 词嵌入层
        self.embedding = nn.Embedding(len(vocab), input_dim, padding_idx=vocab["<pad>"])

        # 位置编码层
        self.position_encoding = PositionalEncoding(input_dim, max_len)

        # Transformer编码器层 (作为Decoder使用)
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类层
        self.classify = nn.Linear(input_dim, len(vocab))
        self.loss = nn.functional.cross_entropy

    # 生成因果注意力掩码 (确保模型只能看到当前位置之前的信息)
    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        device = x.device
        seq_len = x.size(1)

        # 词嵌入
        x_embed = self.embedding(x)

        # 添加位置编码
        x = self.position_encoding(x_embed)

        # 生成因果注意力掩码
        causal_mask = self.generate_causal_mask(seq_len).to(device)

        # 通过Transformer编码器
        x = self.transformer(x, mask=causal_mask)

        # 分类预测
        y_pred = self.classify(x)

        if y is not None:
            # 计算损失 - 只计算最后一个token的损失
            return self.loss(y_pred.view(-1, self.vocab_size), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<UNK>": 1}
    index = 2  # 从2开始，因为0和1已被占用
    with open(vocab_path, encoding="utf8") as f:
        for line in f:
            char = line.strip()  # 使用strip()而不是[:-1]，避免换行符问题
            if char and char not in vocab:  # 确保非空且不在词汇表中
                vocab[char] = index
                index += 1
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            # 只添加非空行
            stripped = line.strip()
            if stripped:
                corpus += stripped
    return corpus


# 随机生成一个样本
def build_sample(vocab, window_size, corpus):
    # 确保有足够的文本生成样本
    if len(corpus) < window_size + 1:
        raise ValueError(f"语料库太短 ({len(corpus)} 字符), 无法生成窗口大小为 {window_size} 的样本")

    start = random.randint(0, len(corpus) - window_size - 1)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    # 安全地将字符转换为索引
    def safe_char_to_idx(char):
        return vocab.get(char, vocab["<UNK>"])

    x = [safe_char_to_idx(char) for char in window]
    y = [safe_char_to_idx(char) for char in target]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        try:
            x, y = build_sample(vocab, window_size, corpus)
            dataset_x.append(x)
            dataset_y.append(y)
        except ValueError as e:
            print(f"生成样本时出错: {e}")
            continue

    # 检查是否有有效的样本
    if not dataset_x:
        raise RuntimeError("无法生成任何有效样本，请检查语料库大小和窗口大小")

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = {idx: char for char, idx in vocab.items()}
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过50字则终止迭代
        while pred_char != "\n" and len(openings) <= 50:
            openings += pred_char
            # 只取最后window_size个字符作为输入
            input_chars = openings[-window_size:]
            x = [vocab.get(char, vocab["<UNK>"]) for char in input_chars]

            # 确保输入长度正确
            if len(x) < window_size:
                # 填充到正确长度
                x = [vocab["<pad>"]] * (window_size - len(x)) + x

            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()

            # 安全地获取预测
            y_pred = model(x)
            if y_pred.dim() == 3:
                y_pred = y_pred[0][-1]
            else:
                y_pred = y_pred[-1]

            index = sampling_strategy(y_pred)

            # 确保索引在有效范围内
            if index >= len(reverse_vocab):
                index = vocab["<UNK>"]

            pred_char = reverse_vocab.get(index, "<UNK>")
    return openings


def sampling_strategy(prob_distribution):
    # 确保概率分布有效
    if torch.isnan(prob_distribution).any():
        # 如果出现NaN，使用均匀分布
        prob_distribution = torch.ones_like(prob_distribution) / len(prob_distribution)

    # 转换为numpy前移到CPU
    prob_distribution = prob_distribution.cpu().numpy()

    # 确保概率和为1
    if np.sum(prob_distribution) <= 0:
        prob_distribution = np.ones_like(prob_distribution) / len(prob_distribution)
    else:
        prob_distribution = prob_distribution / np.sum(prob_distribution)

    if random.random() > 0.1:
        # 贪婪策略
        return np.argmax(prob_distribution)
    else:
        # 采样策略 - 应用温度参数增加多样性
        temperature = 0.8
        scaled_probs = np.exp(np.log(prob_distribution) / temperature)
        scaled_probs = scaled_probs / np.sum(scaled_probs)
        return np.random.choice(len(scaled_probs), p=scaled_probs)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    if len(sentence) < 2:
        return float('inf')  # 无法计算短句子的困惑度

    total_log_prob = 0.0
    valid_chars = 0
    model.eval()

    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]

            # 填充到窗口大小
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x

            x = torch.LongTensor([x])
            target_char = sentence[i]
            target_index = vocab.get(target_char, vocab["<UNK>"])

            if torch.cuda.is_available():
                x = x.cuda()

            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index].item()

            # 避免log(0)的情况
            if target_prob < 1e-10:
                target_prob = 1e-10

            total_log_prob += math.log(target_prob, 2)  # 使用log2计算
            valid_chars += 1

    if valid_chars == 0:
        return float('inf')

    avg_log_prob = total_log_prob / valid_chars
    perplexity = 2 ** (-avg_log_prob)
    return perplexity


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 20  # 样本文本长度（增加窗口大小）

    # 确保语料库足够大
    vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)

    if len(corpus) < window_size * 10:
        print(f"警告：语料库可能太小 ({len(corpus)} 字符)，建议至少是窗口大小的10倍")

    model = build_model(vocab, char_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    # 使用较小的学习率和权重衰减
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    print("文本词表模型加载完毕，开始训练")
    print(f"语料库长度: {len(corpus)} 字符")
    print(f"词汇表大小: {len(vocab)}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 计算实际可用的批次数量
        actual_batches = min(int(train_sample / batch_size), int(len(corpus) / window_size / batch_size))
        if actual_batches == 0:
            print("错误：没有足够的语料生成样本")
            break

        for batch in range(actual_batches):
            try:
                x, y = build_dataset(batch_size, vocab, window_size, corpus)
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()

                optim.zero_grad()
                loss = model(x, y)
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optim.step()
                watch_loss.append(loss.item())

                # 每100个batch打印一次进度
                if batch % 100 == 0:
                    avg_loss = np.mean(watch_loss[-100:]) if len(watch_loss) > 100 else np.mean(watch_loss)
                    print(f"Epoch {epoch + 1}, Batch {batch}/{actual_batches}, Loss: {avg_loss:.4f}")
            except Exception as e:
                print(f"训练批次 {batch} 时出错: {e}")
                continue

        # 更新学习率
        scheduler.step()

        avg_loss = np.mean(watch_loss) if watch_loss else float('nan')
        print("=" * 50)
        print(f"第{epoch + 1}轮平均loss: {avg_loss:.4f}")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 20)

        # 生成示例
        try:
            print("生成示例1:", generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
            print("生成示例2:", generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
        except Exception as e:
            print(f"生成示例时出错: {e}")

        print("=" * 50)

        # 计算训练集的困惑度
        if len(corpus) > 100:
            sample_text = corpus[:100]  # 取前100个字符计算困惑度
            ppl = calc_perplexity(sample_text, model, vocab, window_size)
            print(f"训练集样本困惑度: {ppl:.2f}")

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'config': {
                'input_dim': char_dim,
                'window_size': window_size
            }
        }, model_path)
        print(f"模型保存至: {model_path}")
        return


if __name__ == "__main__":
    train("corpus.txt", False)
