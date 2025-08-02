#coding:utf8

import torch
import json
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
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    # #当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, y=None):
    #     if y is not None:
    #         #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
    #         mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
    #         # print(mask, mask.shape)
    #         if torch.cuda.is_available():
    #             mask = mask.cuda()
    #         x, _ = self.bert(x, attention_mask=mask)
    #         y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
    #         return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
    #     else:
    #         #预测时，可以不使用mask
    #         x, _ = self.bert(x)
    #         y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
    #         return torch.softmax(y_pred, dim=-1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        logits = self.classify(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        return logits

#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料

def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
                corpus.append(json.loads(line))  # 逐行解析
    return corpus
# corpus = load_corpus(r"F:\BaiduNetdiskDownload\week11大语言模型相关第一讲\training_data.json")
def build_sample(tokenizer, sample, max_length=128):
    text = f"指令：{sample['instruction']}输入：{sample['input']}回答：{sample['output']}"
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return inputs["input_ids"], inputs["attention_mask"]
# def load_corpus(path):
#     corpus = ""
#     with open(path, encoding="gbk") as f:
#         for line in f:
#             corpus += line.strip()
#     return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
# def build_sample(tokenizer, window_size, corpus):
#     start = random.randint(0, len(corpus) - 1 - window_size)
#     end = start + window_size
#     window = corpus[start:end]
#     target = corpus[start + 1:end + 1]  #输入输出错开一位
#
#     x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   #将字转换成序号
#     y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
#
#     return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(batch_size, tokenizer, corpus, max_length=128):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for _ in range(batch_size):
        sample = random.choice(corpus)

        # 编码输入部分（instruction + input）
        input_text = f"指令：{sample['instruction']}\n输入：{sample['input']}"
        input_enc = tokenizer(
            input_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 编码输出部分（output）
        output_enc = tokenizer(
            sample['output'],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids_list.append(input_enc["input_ids"])
        attention_mask_list.append(input_enc["attention_mask"])
        labels_list.append(output_enc["input_ids"])

    return (
        torch.cat(input_ids_list),
        torch.cat(attention_mask_list),
        torch.cat(labels_list)
    )

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            inputs = tokenizer(openings, return_tensors="pt", add_special_tokens=False)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = model(**inputs)
            prob_dist = torch.softmax(logits[0, -1], dim=-1)  # 添加softmax转换
            index = sampling_strategy(prob_dist)
            pred_char = tokenizer.decode([index])
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
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)



def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    

    pretrain_model_path = r'F:\BaiduNetdiskDownload\sim_bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            input_ids, attention_mask, labels = build_dataset(batch_size, tokenizer, corpus)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            optim.zero_grad()
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"F:\BaiduNetdiskDownload\week11大语言模型相关第一讲\training_data.json", False)
