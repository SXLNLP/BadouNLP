#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer,BertModel
"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self,bert_path):
        super(LanguageModel, self).__init__()

        self.encoder=BertModel.from_pretrained(bert_path,return_dict=False)
        self.classify = nn.Linear(768, 21128)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, y=None):
    #     x, _ = self.encoder(x)        #output shape:(batch_size, sen_len, input_dim)
    #     y_pred = self.classify(x)   #output shape:(batch_size, sen_len,vocab_size)
    #
    #
    #     if y is not None:
    #         return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
    #     else:
    #
    #         return torch.softmax(y_pred, dim=-1)
    def forward(self, x, y=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.encoder(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.encoder(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip().replace(" ", "")
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus,model,toki):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    x=toki.encode(window,add_special_tokens=False,max_length=window_size, truncation=True,padding="max_length")
    y=toki.encode(target,add_special_tokens=False,max_length=window_size, truncation=True,padding="max_length")
    return x, y

def build_dataset(sample_length, window_size, corpus,model,toki):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample( window_size, corpus,model,toki)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型


# 文本生成测试代码
def generate_sentence(openings, model, window_size,tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x= tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]#y->vocab_size
            index = sampling_strategy(y)
            pred_char = tokenizer.decode(index).replace(" ","").replace("#","")
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
#计算文本ppl


def train(corpus_path, save_weight):
    epoch_num = 20       #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    learning_rate = 0.0001
    corpus = load_corpus(corpus_path)     #加载语料
    bert_path = r"C:\\Users\\晋晨曦\\PycharmProjects\\nlp_codes\\week06\\bert-base-chinese"

    model = LanguageModel(bert_path)    #建立模型
    tokenizer=BertTokenizer.from_pretrained(bert_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size,window_size, corpus,model,tokenizer) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("他在半年之前，就不能做出", model,  window_size,tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, window_size,tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        torch.save(model.state_dict(), base_name)
        return

if __name__ == "__main__":
    train("corpus.txt", False)
