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
基于pytorch的Bert语言模型
"""

# bert_path = r"D:\www.root\bert-base-chinese"

class LanguageModel(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        """ 使用bert """
        self.layer = BertModel.from_pretrained(bert_path, return_dict=False)
        """ 线形层的输入维度，使用bert的hidden_size """
        hidden_size = self.layer.config.hidden_size
        self.classify = nn.Linear(hidden_size, self.layer.config.vocab_size)

        self.dropout = nn.Dropout(0.1)#0.1
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):

        # y_pred = self.dropout(y_pred)
        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            x, _ = self.layer(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)

            y_pred_2 = y_pred.view(-1, y_pred.shape[-1])
            y_2 = y.view(-1)
            return self.loss(y_pred_2, y_2)
        else:
            x, _ = self.layer(x)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)
            return torch.softmax(y_pred, dim=-1) #dim可以不赋值，默认就是-1，代表使用最后一个维度

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

#随机生成一个样本（使用自定义词表，本示例代码是基于bert的，用不到此方法）
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

#随机生成一个样本（使用bert）
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample_with_bert(window_size, corpus, tokenizer):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位

    # print(window, target)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]

    """ 
        使用bert的encode
        ValueError: expected sequence of length 12 at dim 1 (got 11) 
    """
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少。相当于是batch_size
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, window_size, corpus, tokenizer):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample_with_bert(window_size, corpus, tokenizer)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(bert_path):
    model = LanguageModel(bert_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, window_size, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y_pred = model(x)
            y = y_pred[0][-1]
            index = sampling_strategy(y)
            # pred_char = reverse_vocab[index
            pred_char = "".join(tokenizer.decode(index))
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        #贪婪解码，取最大概率的一个选中
        strategy = "greedy"
    else:
        # 随机采样，概率越大的，选中几率最大
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        # p参数： 1D 数组，可选。表示每个元素被选中的概率。该数组的长度应与 a 的长度相同，且所有概率之和应为 1。如果没有指定 p，则所有元素的概率相同。
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


def train(corpus_path, bert_path, save_weight=True):
    epoch_num = 20        #训练轮数 20
    batch_size = 128       #每次训练样本个数 64
    train_sample = 10000   #每轮训练总共训练的样本总数 50000
    # char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    # vocab = build_vocab(bert_vocab_path)       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(bert_path)    #建立模型

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus, tokenizer) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, window_size, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, window_size, tokenizer))
    if not save_weight:
        return
    else:
        # base_name = os.path.basename(corpus_path).replace("txt", "pth")
        # model_path = os.path.join("model", base_name)
        model_path = os.path.join("output", "nnlm_bert.pth")
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    bert_path = r"D:\www.root\bert-base-chinese"
    train("corpus.txt", bert_path)
