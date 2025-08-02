#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

# 新增：引入transformers库
from transformers import BertTokenizer
from transformers import BertModel

"""
基于pytorch的LSTM语言模型
可选BERT作为encoder
"""

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab, encoder_type='lstm', bert_name=(os.path.dirname(os.path.abspath(__file__))+"/../../models/bert-base-chinese")):
        super(LanguageModel, self).__init__()
        print("BERT模型加载路径：", bert_name)
        self.encoder_type = encoder_type
        self.vocab = vocab
        if encoder_type == 'lstm':
            self.embedding = nn.Embedding(len(vocab), input_dim)
            self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
            self.classify = nn.Linear(input_dim, len(vocab))
            self.dropout = nn.Dropout(0.1)
        elif encoder_type == 'bert':
            self.bert_name = bert_name
            self.bert = BertModel.from_pretrained(os.path.dirname(os.path.abspath(__file__))+"/../../models/bert-base-chinese", return_dict=True)

            # self.bert = BertModel.from_pretrained(bert_name)
            self.tokenizer = BertTokenizer.from_pretrained(bert_name)
            self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if self.encoder_type == 'lstm':
            x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
            x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        elif self.encoder_type == 'bert':
            # x: (batch, seq_len) 的token id，需转为字符串再用BERT分词器
            # 这里假设x是原始token id序列（与vocab对应），需转为文本
            batch_text = []
            reverse_vocab = dict((y, x) for x, y in self.vocab.items())
            for seq in x:
                chars = [reverse_vocab.get(int(idx), '[UNK]') for idx in seq]
                batch_text.append(''.join(chars))
            # 用BERT分词器编码
            bert_inputs = self.tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)
            if next(self.parameters()).is_cuda:
                bert_inputs = {k: v.cuda() for k, v in bert_inputs.items()}
            outputs = self.bert(**bert_inputs)
            x = outputs.last_hidden_state  # (batch, seq_len, hidden)
            y_pred = self.classify(x)      # (batch, seq_len, vocab_size)
        if y is not None:
            if self.encoder_type == 'bert':
                # target 直接用BERT分词器分出来的token id
                target = bert_inputs['input_ids']
                return self.loss(y_pred.view(-1, y_pred.shape[-1]), target.view(-1))
            else:
                return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

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

#随机文本训练样本
#从文本中截取随机窗口，窗口大小为window_size，进行并行训练，目标样本是输入样本向左移动一位。
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

#将训练样本组成数据集。
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
# 新增encoder_type参数
def build_model(vocab, char_dim, encoder_type='lstm'):
    model = LanguageModel(char_dim, vocab, encoder_type=encoder_type)
    return model

#文本生成测试代码
#openings 引言 字符串
# 新增encoder_type参数
def generate_sentence(openings, model, vocab, window_size, encoder_type='lstm'):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char #不断叠加预测的字符转位输入字符串
            if encoder_type == 'lstm':
                x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
                x = torch.LongTensor([x])
            elif encoder_type == 'bert':
                # 直接用BERT分词器编码
                x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
                x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x) #[1,window_size, len(vocab)]
            y = y[0][-1] #每次取最后一个字符的len(vocab)维度向量
            index = sampling_strategy(y)#将向量转为字符索引
            pred_char = reverse_vocab[index]
    return openings

#采样策略
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy" #贪婪采样greedy
    else:
        strategy = "sampling" #概率采样

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
            pred_prob_distribute = model(x)
            pred_prob_distribute = pred_prob_distribute[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


#训练流程
# 新增encoder_type参数
def train(corpus_path, save_weight=True, encoder_type='lstm'):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    vocab = build_vocab(os.path.dirname(os.path.abspath(__file__))+"/vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim, encoder_type=encoder_type)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size, encoder_type=encoder_type))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size, encoder_type=encoder_type))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    # 传入encoder_type='bert'即可用BERT做encoder
    train(os.path.dirname(os.path.abspath(__file__))+"/corpus.txt", False, encoder_type='bert')
