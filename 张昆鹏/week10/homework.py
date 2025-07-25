#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, model_name):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        attention_mask = (x != self.tokenizer.pad_token_id).long()
        seq_len = x.size(1)

        # 构造下三角矩阵
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(1) * causal_mask  
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        logits = self.classify(outputs[0])

        if y is not None:
            return self.loss(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            return torch.softmax(logits, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    while True:
        start = random.randint(0, len(corpus) - 1 - window_size - 1)
        text = corpus[start:start + window_size + 1]  # 多取1位，用于构造 y
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == window_size + 1:
            x = tokens[:window_size]
            y = tokens[1:]
            return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    max_len = max(len(x) for x in dataset_x)
    pad_id = tokenizer.pad_token_id
    for i in range(len(dataset_x)):
        while len(dataset_x[i]) < max_len:
            dataset_x[i].append(pad_id)
            dataset_y[i].append(pad_id)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(model_name):
    model = LanguageModel(model_name)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size, max_len=30):
    model.eval()
    generated = tokenizer.encode(openings, add_special_tokens=False)
    device = next(model.parameters()).device
    with torch.no_grad():
        pred_char = ""
        while len(generated) < max_len:
            input_ids = generated[-window_size:]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            outputs = model(input_tensor)
            logits = outputs[:, -1, :]

            next_token_id = sampling_strategy(logits[0])
            generated.append(next_token_id)

            next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]
            if next_token in ["\n", tokenizer.sep_token, tokenizer.eos_token]:
                break
    tokens = tokenizer.convert_ids_to_tokens(generated)
    text = "".join([t.replace("##", "") for t in tokens if t not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]])
    return text

def sampling_strategy(logits, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
    if random.random() < 0.1:
        # random
        return int(np.random.choice(len(probs), p=probs))
    else:
        # greddy
        return int(np.argmax(probs))

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
    batch_size = 256       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    window_size = 20       #样本文本长度
    model_name = r'N:\八斗\上一期\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(model_name)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)   #建立优化器
    print("BERT加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample // batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"N:\八斗\上一期\第十周 文本生成\week10 文本生成问题\lstm语言模型生成文本\corpus.txt", False)
