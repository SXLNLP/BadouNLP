#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertConfig

"""
基于pytorch的LSTM语言模型
"""

PAD = 0
MASK = 1
UNK = 2

class BertLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4, max_len=512):
        super(BertLanguageModel, self).__init__()
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_len,
            pad_token_id=PAD
        )
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD)
        self.bert = BertModel(self.config, add_pooling_layer=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        mask = (x != PAD).long()
        bert_out = self.bert(
            input_ids=x,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(x)
        ).last_hidden_state
        logits = self.lm_head(bert_out)   #output shape:(batch_size, seq, vocab_size)
        if y is not None:
            return self.loss(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            return torch.softmax(logits, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":PAD, "<mask>":MASK, "<unk>":UNK}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()       #去掉结尾换行符
            vocab[char] = index + 3 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def random_mask(tokens, vocab, mask_prob=0.15):
    """
    对输入序列做MLM的随机 mask：
    80%替换成[MASK]，10%随机词，10%不变
    返回 masked_tokens 和 对应的 labels（-100表示不计算loss）
    """
    labels = tokens[:]
    for i in range(len(tokens)):
        prob = random.random()
        if prob < mask_prob:
            prob /= mask_prob
            if prob < 0.8:
                tokens[i] = vocab["<mask>"]
            elif prob < 0.9:
                tokens[i] = random.choice(list(vocab.values()))
            labels[i] = labels[i]
        else:
            labels[i] = -100
    return tokens, labels

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, seq_len, corpus):
    start = random.randint(0, len(corpus) - seq_len)
    chunk = corpus[start:start + seq_len]
    tokens = [vocab.get(c, UNK) for c in chunk]
    masked_tokens, labels = random_mask(tokens, vocab)
    return masked_tokens, labels

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_num, vocab, seq_len, corpus):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_num):
        x, y = build_sample(vocab, seq_len, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab_size, hidden=256):
    model = BertLanguageModel(vocab_size, hidden)
    return model

#文本生成测试代码
def generate_sentence(prompt, model, vocab, max_len=30):
    """
    自回归逐字生成，直到遇到换行或达到 max_len。
    prompt : 初始字符串，如 "让他在半年之前，就不能做出"
    """
    model.eval()
    device = next(model.parameters()).device
    reverse_vocab = {v: k for k, v in vocab.items()}
    result = prompt

    with torch.no_grad():
        while len(result) < max_len:
            # 1. 把当前已生成文本转成 token id，长度不够就整段输入
            token_ids = [vocab.get(c, vocab["<unk>"]) for c in result[-max_len:]]
            x = torch.LongTensor([token_ids]).to(device)

            # 2. 构造 causal mask：下三角 1
            seq_len = x.size(1)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).to(device)
            pad_mask = (x != vocab["<pad>"]).long()
            attention_mask = causal_mask * pad_mask.unsqueeze(1)

            # 3. 前向，只取最后一个位置的 logits
            bert_out = model.bert(input_ids=x, attention_mask=attention_mask).last_hidden_state
            logits = model.lm_head(bert_out[0, -1, :])     # [vocab]
            prob = torch.softmax(logits, dim=-1)

            # 4. 采样下一个字符
            next_id = sampling_strategy(prob)
            next_char = reverse_vocab[next_id]

            if next_char == "\n":
                break
            result += next_char
    return result

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(len(prob_distribution), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    hidden_size = 256        #每个字的维度
    seq_len = 64       #样本文本长度
    vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(len(vocab), hidden_size)    #建立模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, seq_len, corpus) #构建一组训练样本
            x, y = x.to(device), y.to(device)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
