# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import re
from transformers import BertConfig, BertModel

"""
基于BERT的语言模型（含三角形Mask，用于自回归文本生成）
"""


class BertLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=6, num_heads=12, max_seq_len=512):
        super(BertLanguageModel, self).__init__()
        # 配置BERT参数
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_seq_len,
            type_vocab_size=1,  # 单句任务
            is_decoder=True  # 标记为解码器，支持自回归生成
        )
        # 初始化BERT模型（使用解码器模式，支持自回归）
        self.bert = BertModel(self.config)
        # 语言模型头：将隐藏状态映射到词表
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        # 损失函数
        self.loss = nn.functional.cross_entropy
        # 最大序列长度（用于生成三角形Mask）
        self.max_seq_len = max_seq_len

    def generate_triangle_mask(self, seq_len):
        """生成三角形掩码（下三角掩码）：防止关注未来信息"""
        # 生成(seq_len, seq_len)的下三角矩阵，对角线及以下为0（允许关注），以上为-∞（禁止关注）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # 上三角为1，下三角为0
        mask = mask.masked_fill(mask == 1, float('-inf'))  # 上三角填充为-∞
        return mask  # 形状：(seq_len, seq_len)

    def forward(self, x, y=None):
        # 生成三角形掩码（适配输入序列长度）
        seq_len = x.shape[1]
        triangle_mask = self.generate_triangle_mask(seq_len).to(x.device)  # 移到与输入相同的设备

        # BERT前向传播（传入三角形掩码）
        outputs = self.bert(
            input_ids=x,
            attention_mask=None,  # 不使用padding掩码（简化版）
            decoder_attention_mask=triangle_mask  # 应用三角形掩码（自回归掩码）
        )
        sequence_output = outputs.last_hidden_state  # 形状：(batch_size, seq_len, hidden_dim)
        logits = self.lm_head(sequence_output)  # 形状：(batch_size, seq_len, vocab_size)

        if y is not None:
            # 计算损失：只对有效位置计算（y中不为-100的位置）
            return self.loss(logits.view(-1, logits.shape[-1]), y.view(-1))
        else:
            # 推理时返回概率分布
            return torch.softmax(logits, dim=-1)


# 加载字表（添加BERT所需特殊token）
def build_vocab(vocab_path):
    vocab = {
        "[PAD]": 0,  # 填充token
        "[CLS]": 1,  # 句子起始标记
        "[SEP]": 2,  # 句子分隔标记
        "[MASK]": 3,  # 掩码标记
        "[UNK]": 4  # 未知token
    }
    with open(vocab_path, encoding="utf8") as f:
        for line in f:
            char = line.strip()  # 去掉换行符
            if char and char not in vocab:
                vocab[char] = len(vocab)  # 按顺序添加字符
    reverse_vocab = {v: k for k, v in vocab.items()}  # 反向映射：ID→字符
    return vocab, reverse_vocab


# 加载语料（保持原逻辑，略作调整）
def load_corpus(path):
    corpus = []
    with open(path, encoding="gbk") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                corpus.append(line)
    return corpus  # 返回句子列表，而非单一字符串


# 随机生成样本（保持滑动窗口预测下一个字符的逻辑）
def build_sample(vocab, window_size, corpus):
    # 随机选择一个句子
    sentence = random.choice(corpus)
    # 确保句子长度足够
    if len(sentence) < window_size + 1:
        return build_sample(vocab, window_size, corpus)  # 重新采样
    # 随机选择起始位置
    start = random.randint(0, len(sentence) - window_size - 1)
    end = start + window_size
    # 输入窗口：[start:end]，目标窗口：[start+1:end+1]（错开一位，预测下一个字符）
    input_window = sentence[start:end]
    target_window = sentence[start + 1:end + 1]
    # 转换为ID（未知字符用[UNK]）
    x = [vocab.get(char, vocab["[UNK]"]) for char in input_window]
    y = [vocab.get(char, vocab["[UNK]"]) for char in target_window]
    return x, y


# 建立数据集（批量生成样本）
def build_dataset(sample_num, vocab, window_size, corpus):
    dataset_x, dataset_y = [], []
    for _ in range(sample_num):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    # 转换为LongTensor
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 构建模型
def build_model(vocab_size):
    model = BertLanguageModel(vocab_size)
    return model


# 文本生成函数（适配BERT模型）
def generate_sentence(openings, model, vocab, reverse_vocab, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成终止条件：遇到换行或长度超30
        while pred_char != "\n" and len(openings) <= 30:
            # 更新输入序列：添加预测字符，截取最近window_size个字符
            current_input = openings + pred_char
            input_ids = [vocab.get(char, vocab["[UNK]"]) for char in current_input[-window_size:]]
            # 转换为张量并添加批次维度
            input_tensor = torch.LongTensor([input_ids]).to(next(model.parameters()).device)
            # 模型预测
            pred_probs = model(input_tensor)[0][-1]  # 取最后一个位置的概率分布
            # 采样策略（保持原逻辑：90%贪心，10%随机）
            index = sampling_strategy(pred_probs)
            pred_char = reverse_vocab[index]
        # 返回生成结果（去掉初始空字符）
        return openings + pred_char


# 采样策略（保持原逻辑）
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:  # 90%概率贪心
        return torch.argmax(prob_distribution).item()
    else:  # 10%概率随机采样
        prob_np = prob_distribution.cpu().numpy()
        return np.random.choice(len(prob_np), p=prob_np)


# 计算困惑度（保持原逻辑）
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            # 取前i个字符作为输入，预测第i个字符
            start = max(0, i - window_size)
            input_window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in input_window]
            x_tensor = torch.LongTensor([x]).to(next(model.parameters()).device)
            # 目标字符
            target_char = sentence[i]
            target_id = vocab.get(target_char, vocab["[UNK]"])
            # 预测概率
            pred_probs = model(x_tensor)[0][-1]  # 最后一个位置的概率分布
            target_prob = pred_probs[target_id].item()
            prob += math.log(target_prob + 1e-10, 10)  # 加微小值避免log(0)
    # 计算PPL：2^(-平均对数概率)
    return 2 ** (-prob / len(sentence))


# 训练函数
def train(corpus_path, save_weight=True):
    # 超参数配置
    epoch_num = 20
    batch_size = 32  # BERT参数量大，减小batch_size
    train_sample = 20000  # 每轮样本数（BERT训练更耗时）
    window_size = 10
    hidden_dim = 768  # BERT基础版隐藏维度
    num_layers = 6  # BERT基础版层数
    num_heads = 12  # BERT基础版注意力头数

    # 加载词表和语料
    vocab, reverse_vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    print(f"词表大小：{len(vocab)}，语料句子数：{len(corpus)}")

    # 构建模型
    model = build_model(len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器（BERT常用AdamW，学习率更小）
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # 训练循环
    print(f"模型加载到{device}，开始训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = []
        # 每轮生成多个batch
        for batch_idx in range(train_sample // batch_size):
            # 生成批次数据
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            x, y = x.to(device), y.to(device)
            # 梯度清零
            optim.zero_grad()
            # 计算损失
            loss = model(x, y)
            # 反向传播
            loss.backward()
            # 更新参数
            optim.step()
            # 记录损失
            total_loss.append(loss.item())
            # 打印批次进度
            if (batch_idx + 1) % 50 == 0:
                print(f"批次 {batch_idx + 1}/{train_sample // batch_size}，当前loss：{loss.item():.4f}")

        # 每轮结束后评估
        avg_loss = np.mean(total_loss)
        print(f"\n========= 第{epoch + 1}轮 =========")
        print(f"平均loss：{avg_loss:.4f}")
        # 生成示例文本
        print("生成示例1：", generate_sentence("让他在半年之前，就不能做出", model, vocab, reverse_vocab, window_size))
        print("生成示例2：", generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, reverse_vocab, window_size))

    # 保存模型
    if save_weight:
        os.makedirs("model", exist_ok=True)
        model_path = os.path.join("model", "bert_lm.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型保存至：{model_path}")


if __name__ == "__main__":
    train("corpus.txt", False)
