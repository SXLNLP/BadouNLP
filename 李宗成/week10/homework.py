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
基于bert模型实现文本生成
"""

def create_mask(input_ids):
    # 保持原始形状
    batch_size, seq_length = input_ids.shape
    
    # 创建上三角attention mask
    # tril表示保留下三角（包括对角线），这样每个token只能看到自己和之前的token
    causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device))

    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    return causal_mask

class LanguageModel(nn.Module):
    def __init__(self, bert_model_path):
        super(LanguageModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        attention_mask = create_mask(x)
        
        x, _ = self.bert(x, attention_mask=attention_mask)
        
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)
#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    return window, target

#corpus 语料字符串
def build_dataset(tokenizer, sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus)
        x = tokenizer.encode(
            x, 
            add_special_tokens=True, 
            return_tensors="pt",
            padding='max_length',
            max_length=100, 
            truncation=True
        )
        y = tokenizer.encode(
            y, 
            add_special_tokens=True, 
            return_tensors="pt",
            padding='max_length',
            max_length=100,
            truncation=True
        )
        dataset_x.append(x.squeeze().tolist())
        dataset_y.append(y.squeeze().tolist())
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(bert_model_path):
    model = LanguageModel(bert_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = model.tokenizer.encode(openings[-window_size:], add_special_tokens=True, return_tensors="pt")
            #x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            # 生成字
            pred_char = model.tokenizer.decode([index])
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
    batch_size = 16       #每次训练样本个数
    train_sample = 3200   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    bert_model_path = r"D:\\workspace\\pretrain_models\\bert-base-chinese"
    # vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(bert_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.2e-5)   #建立优化器


    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(model.tokenizer, batch_size, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, window_size))
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
