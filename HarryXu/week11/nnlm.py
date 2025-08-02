#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertModel

"""
基于pytorch的BERT语言模型，支持SFT训练（处理标题-内容结构的JSON数据）
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def generate_square_subsequent_mask(self, sz):
        # 生成上三角矩阵（对角线及以下为1，其余为0）用于自回归
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 创建自回归掩码
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        bert_output = self.bert(x).last_hidden_state  # [batch_size, seq_len, hidden_size]
        y_pred = self.classify(self.dropout(bert_output))   # output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            masked_y_pred = y_pred.masked_fill(mask.unsqueeze(-1) == float('-inf'), 0)
            return self.loss(masked_y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<unk>": 1}  # 添加unk token
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       # 去掉结尾换行符
            vocab[char] = index + 2  # 留出0位给pad token，1位给unk token
    return vocab

# 从JSON文件加载SFT训练数据
def load_sft_data(json_path):
    """加载标题-内容结构的JSON数据"""
    sft_data = []
    with open(json_path, encoding="utf8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # 组合标题和内容，添加分隔符以便模型学习结构
                combined = f"标题：{item['title']} 内容：{item['content']}"
                sft_data.append(combined)
            except Exception as e:
                print(f"解析JSON出错: {e}，跳过该行")
    return sft_data

# 从SFT数据构建单个样本
def build_sft_sample(vocab, window_size, sft_data):
    # 随机选择一个样本
    sample = random.choice(sft_data)
    # 确保样本长度足够
    if len(sample) <= window_size:
        return build_sft_sample(vocab, window_size, sft_data)
    
    # 随机选择窗口起始位置
    start = random.randint(0, len(sample) - 1 - window_size)
    end = start + window_size
    window = sample[start:end]
    target = sample[start + 1:end + 1]  # 输入输出错开一位
    
    # 将字转换成序号
    x = [vocab.get(word, vocab["<unk>"]) for word in window]
    y = [vocab.get(word, vocab["<unk>"]) for word in target]
    return x, y

# 建立SFT数据集
def build_sft_dataset(sample_length, vocab, window_size, sft_data):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sft_sample(vocab, window_size, sft_data)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过100字则终止迭代
        while pred_char != "\n" and len(openings) <= 100:
            openings += pred_char
            # 取最近的window_size个字符作为输入
            x = [vocab.get(char, vocab["<unk>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            # 获取预测分布
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab.get(index, "<unk>")
    return openings

def sampling_strategy(prob_distribution):
    # 混合使用贪婪采样和随机采样
    if random.random() > 0.1:  # 90%概率贪婪采样，10%概率随机采样
        return int(torch.argmax(prob_distribution))
    else:
        prob_distribution = prob_distribution.cpu().numpy()
        # 增加温度参数调整随机性
        temperature = 0.7
        prob_distribution = np.exp(np.log(prob_distribution) / temperature)
        prob_distribution = prob_distribution / np.sum(prob_distribution)
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    if len(sentence) < 2:
        return float('inf')
        
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<unk>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<unk>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(max(target_prob.item(), 1e-10), 10)  # 防止log(0)
    return 2 **(prob * (-1 / len(sentence)))


def train_sft(json_path, save_weight=True):
    epoch_num = 30         # 训练轮数，SFT通常需要更多轮次
    batch_size = 32        # 每次训练样本个数，BERT模型通常用较小的batch size
    train_sample = 30000   # 每轮训练总共训练的样本总数
    char_dim = 768         # BERT隐藏层维度
    window_size = 32       # 样本文本长度，可适当增大
    vocab = build_vocab("vocab.txt")       # 建立字表
    sft_data = load_sft_data(json_path)    # 加载SFT数据
    
    if not sft_data:
        print("没有加载到有效的训练数据，请检查JSON文件")
        return
    
    print(f"加载SFT数据完成，共{len(sft_data)}条样本")
    
    model = build_model(vocab, char_dim)    # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU进行训练")
    else:
        print("使用CPU进行训练")
    
    # 优化器设置，SFT通常使用较小的学习率
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epoch_num)
    
    print("模型加载完毕，开始SFT训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_sft_dataset(batch_size, vocab, window_size, sft_data)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
            
            # 打印批次进度
            if (batch + 1) % 50 == 0:
                print(f"第{epoch+1}轮，第{batch+1}批，当前loss: {loss.item():.4f}")
        
        scheduler.step()  # 调整学习率
        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss: {avg_loss:.4f}")
        
        # 每轮训练后生成一些文本进行测试
        print("\n生成测试1:")
        print(generate_sentence("标题：", model, vocab, window_size))
        print("\n生成测试2:")
        print(generate_sentence("标题：超市惊现奇葩事件 内容：", model, vocab, window_size))
        
        # 计算一个示例的困惑度
        if sft_data:
            sample_text = sft_data[0][:100]  # 取第一个样本的前100字符
            ppl = calc_perplexity(sample_text, model, vocab, window_size)
            print(f"\n示例文本困惑度: {ppl:.2f}")
    
    if save_weight:
        if not os.path.exists("sft_model"):
            os.makedirs("sft_model")
        base_name = os.path.basename(json_path).replace("json", "pth")
        model_path = os.path.join("sft_model", base_name)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")
    return


if __name__ == "__main__":
    # 运行SFT训练，使用JSON格式的训练数据
    train_sft("sample_data.json", save_weight=True)
