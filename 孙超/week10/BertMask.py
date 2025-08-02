#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertTokenizer, BertConfig, BertLMHeadModel

"""
基于pytorch + transformers 的 BERT 语言模型
使用BERT作为自回归语言模型
"""

MODEL_NAME = "bert-base-chinese"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 语言模型封装
class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        config = BertConfig.from_pretrained(MODEL_NAME)
        config.is_decoder = True
        config.add_cross_attention = False
        config.vocab_size = vocab_size
        self.model = BertLMHeadModel(config)

    def forward(self, x, y=None):
        outputs = self.model(input_ids=x)
        logits = outputs[0]
        if y is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            return loss
        else:
            return torch.softmax(logits, dim=-1)

# 构建字表
def build_vocab(tokenizer):
    vocab = tokenizer.get_vocab()
    vocab["<pad>"] = tokenizer.pad_token_id
    return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 生成随机样本（输入输出错位）
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - window_size - 1)
    window = corpus[start: start + window_size]
    target = corpus[start + 1: start + window_size + 1]

    x = tokenizer.encode(window, add_special_tokens=False)
    y = tokenizer.encode(target, add_special_tokens=False)

    # padding 到 window_size 长度
    x = x + [tokenizer.pad_token_id] * (window_size - len(x))
    y = y + [tokenizer.pad_token_id] * (window_size - len(y))

    return x, y

# 构建数据集
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x, dataset_y = [], []
    for _ in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab_size):
    model = LanguageModel(vocab_size)
    return model

# 生成文本
def generate_sentence(openings, model, tokenizer, window_size):
    reverse_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "。" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings[-window_size:], add_special_tokens=False)
            x = x + [tokenizer.pad_token_id] * (window_size - len(x))
            x = torch.LongTensor([x]).to(DEVICE)
            y_pred = model(x)[0][-1]
            index = sampling_strategy(y_pred)
            pred_char = tokenizer.decode([index]).replace(" ", "")
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        prob_distribution /= prob_distribution.sum()  # normalize
        return np.random.choice(len(prob_distribution), p=prob_distribution)

# perplexity 计算
def calc_perplexity(sentence, model, tokenizer, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        encoded = tokenizer.encode(sentence, add_special_tokens=False)
        for i in range(1, len(encoded)):
            start = max(0, i - window_size)
            x = encoded[start:i]
            x = x + [tokenizer.pad_token_id] * (window_size - len(x))
            x = torch.LongTensor([x]).to(DEVICE)
            target_index = encoded[i]
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob.item(), 10)
    return 2 ** (-prob / len(sentence))

# 训练主函数
def train(corpus_path, save_weight=True):
    epoch_num = 3
    batch_size = 16
    train_sample = 1000
    window_size = 12
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    vocab = build_vocab(tokenizer)
    corpus = load_corpus(corpus_path)
    model = build_model(len(vocab)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    print("✅ BERT 语言模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if save_weight:
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("model", "bert_lm.pth"))
        print("✅ 模型已保存到 model/bert_lm.pth")
    return

if __name__ == "__main__":
    train("corpus.txt", save_weight=True)
