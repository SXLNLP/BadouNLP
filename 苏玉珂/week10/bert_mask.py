# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertTokenizer, BertForMaskedLM

"""
基于 BERT 的自回归语言模型
"""

class BERTLanguageModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertForMaskedLM.from_pretrained(model_name)
        self.bert.config.is_decoder = True  # 设置为解码器模式，支持自回归
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.loss, outputs.logits

    def generate(self, input_ids, max_length=50, temperature=1.0):
        """
        自回归生成文本
        input_ids: (batch_size, seq_len)
        """
        for _ in range(max_length):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                return_dict=True
            )
            next_token_logits = outputs.logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# 加载字表（BERT 会自动处理）
def build_vocab(model_name='bert-base-chinese'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="utf8") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    # 将文本转换为 token_ids
    input_ids = tokenizer.encode(window, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    # 截断或填充到 window_size
    input_ids = input_ids[:window_size]
    target_ids = target_ids[:window_size]

    # 补齐长度
    input_ids += [tokenizer.pad_token_id] * (window_size - len(input_ids))
    target_ids += [tokenizer.pad_token_id] * (window_size - len(target_ids))

    return input_ids, target_ids

# 建立数据集
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(model_name='bert-base-chinese'):
    model = BERTLanguageModel(model_name)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_token = ""
        openings = tokenizer.encode(openings, add_special_tokens=False)
        openings = torch.LongTensor([openings])
        while pred_token != tokenizer.sep_token and len(openings[0]) <= 30:
            x = openings[:, -window_size:]
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[1][:, -1, :]
            index = torch.argmax(y).item()
            pred_token = tokenizer.decode([index])
            openings = torch.cat([openings, torch.LongTensor([[index]])], dim=1)
        return tokenizer.decode(openings[0], skip_special_tokens=True)

# 计算文本ppl
def calc_perplexity(sentence, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        prob = 0
        for i in range(1, len(tokens)):
            start = max(0, i - window_size)
            window = tokens[start:i]
            x = torch.LongTensor([window])
            target = tokens[i]
            if torch.cuda.is_available():
                x = x.cuda()
            _, logits = model(x)
            target_prob = logits[0, -1, target]
            prob += math.log(target_prob, 10)
        return 2 ** (-prob / len(tokens))

# 训练函数
def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 16
    train_sample = 50000
    window_size = 10
    model_name = 'bert-base-chinese'
    tokenizer = build_vocab(model_name)
    corpus = load_corpus(corpus_path)
    model = build_model(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.AdamW(model.bert.parameters(), lr=2e-5)
    print("模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            loss, _ = model(x, labels=y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.bert.state_dict(), model_path)
        return

if __name__ == "__main__":
    train("corpus.txt", False)
