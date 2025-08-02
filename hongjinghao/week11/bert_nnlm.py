#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertTokenizer, BertModel

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            content_len = len(x)-len(y)
            for i in mask:
                for j in range(0, x.shape[1]):
                    for k in range(0, content_len):
                        i[j][k] = 0

            # print(mask, mask.shape)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

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
    content = []
    title = []
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            content += [line["content"]]
            title += [line["title"]]
    return content, title

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, content, title, i):
    index = i % len(content)
    input = content[index] + title[index]
    output = title[index]
    x = tokenizer.encode(input, add_special_tokens=False, padding='max_length', truncation=True, max_length=100)   #将字转换成序号
    y = tokenizer.encode(output, add_special_tokens=False, padding='max_length', truncation=True, max_length=100)

    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, content, title, sample_index):
    dataset_x = []
    dataset_y = []
    start_index = sample_index % len(content)
    for i in range(start_index, sample_length + start_index):
        x, y = build_sample(tokenizer, content, title, i)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
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
    batch_size = 100    #每次训练样本个数
    train_sample = 500   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    sample_index = 0
    

    pretrain_model_path = r'D:\study\ai\bert-base-chinese\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    content, title = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, content, title, sample_index) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
            sample_index += batch_size
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    if not save_weight:
        return
    else:
        model_path = os.path.join("model", "output.pth")
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    '''
    train(r"D:\study\ai\录播\week11\week11 大语言模型相关第一讲\week11 大语言模型相关第一讲\homework\sample_data.json", True)
    '''
    pretrain_model_path = r'D:\study\ai\bert-base-chinese\bert-base-chinese'
    model = LanguageModel(768, 21128, pretrain_model_path)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    model.load_state_dict(torch.load("model/output.pth"))
    model.eval()
    x = input("请输入content：")
    x = tokenizer.encode(x, add_special_tokens=False, padding='max_length', truncation=True, max_length=100)
    print(x)
    print(torch.LongTensor(x))
    x = torch.LongTensor(x)
    x_batch = x.unsqueeze(0)
    with torch.no_grad():
        output = model(x_batch)
        output = output.squeeze(0)
        output_seq = ""
        for i in range(0, output.shape[0]):
            word_vector = output[i]
            token_id = torch.argmax(word_vector).item()
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            output_seq += token_str
        print(output_seq)