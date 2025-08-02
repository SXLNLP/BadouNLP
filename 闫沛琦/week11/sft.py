#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from transformers import BertModel
from transformers import BertTokenizer
import json
from torch.utils.data import DataLoader

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained("bert-base-chinese")
        self.classify = nn.Linear(input_dim, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            outputs = self.layer(x, attention_mask=mask)
            y_pred = self.classify(outputs.last_hidden_state)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            outputs = self.layer(x)
            y_pred = self.classify(outputs.last_hidden_state)
            return torch.softmax(y_pred, dim=-1)

def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus

def build_dataset(batch_size, tokenizer, window_size, corpus):
    dataset = []
    for i, (question, answer) in enumerate(corpus):
        question_encode = tokenizer.encode(question, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)

        x = [tokenizer.cls_token_id] + question_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(question_encode)*[-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        mask = create_mask(len(question_encode), len(answer_encode))

        x = x[:window_size] + [0] * (max(0, window_size - len(x)))
        y = y[:window_size] + [-1] * (max(0, window_size - len(y)))
        mask = pad_mask(mask, (window_size, window_size))

        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_mask(question_len, answer_len):
    len_question = question_len + 2
    len_answer = answer_len + 1
    mask = torch.ones(len_question+len_answer, len_question+len_answer)
    for i in range(len_question):
        mask[i, len_question:] = 0
    for i in range(len_answer):
        mask[len_question+i, len_question+i+1:] = 0
    return mask

def pad_mask(tensor, target_shape):
    height, width = tensor.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    result[:h_end, :w_end] = tensor[:h_end, :w_end]
    return result

#建立模型
def build_model(vocab_size, char_dim):
    model = LanguageModel(char_dim, vocab_size)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    char_dim = 768        #每个字的维度
    window_size = 50       #样本文本长度
    vocab_size = 21128

    corpus = load_corpus(corpus_path)     #加载语料
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    train_data = build_dataset(batch_size, tokenizer, window_size, corpus) #构建一组训练样本
    model = build_model(vocab_size, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("小学保安用弹“小鸡鸡”手段体罚学生被刑拘", model, tokenizer))
        print(generate_sentence("罗伯斯干扰刘翔是否蓄谋已久？", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    train("sample_data.json", False)
