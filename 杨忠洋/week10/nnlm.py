#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import logging
from transformers import BertModel, BertTokenizer

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
基于pytorch的BERT语言模型
"""
config = {
    "bert_path": r"D:\AI\bert-base-chinese"
}


class LanguageModel(nn.Module):
    def __init__(self, weight, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classify = nn.Linear(self.bert.config.hidden_size, len(vocab))
        self.loss = nn.functional.cross_entropy
        self.weight = weight

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        mask = (x != 0).float()
        bert_output = self.bert(x, attention_mask=mask, return_dict=True, output_hidden_states=True).hidden_states
        sequence_output = bert_output[1]
        y_pred = self.classify(sequence_output)   #output shape:(batch_size, sen_len, hidden_size)
        if y is not None:
            # 添加重复惩罚
            for i in range(1, x.size(1)):
                prev_token = x[:, i - 1]
                y_pred[:, i, prev_token] -= 2.0  # 惩罚系数
            return self.loss(input=y_pred.view(-1, y_pred.shape[-1]),
                             target=y.view(-1),
                             weight=self.weight,
                             ignore_index=0)
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab():
    bert_vocab = BertTokenizer.from_pretrained(config["bert_path"]).vocab
    return bert_vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["[UNK]"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["[UNK]"]) for word in target]

    return x, y

#建立数据集
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
def build_model(vocab, weight):
    model = LanguageModel(weight, vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.7:
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
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def build_weight(vocab, corpus):
    # 统计训练文本中出现的词
    word_freq = {}
    for char in corpus:
        word_freq[char] = word_freq.get(char, 0) + 1
    # 构建权重列表（默认值为 0）
    weights = torch.ones(len(vocab)) * 0.01  # 未出现的词权重较小
    for word, idx in vocab.items():
        if word in word_freq:
            weights[idx] = 1.0  # 出现的词权重为 1
    return weights
def train(corpus_path, save_weight=True):
    epoch_num = 50        #训练轮数
    batch_size = 100       #每次训练样本个数
    train_sample = 20000   #每轮训练总共训练的样本总数
    window_size = 20       #样本文本长度
    vocab = build_vocab()       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    cuda_flag = torch.cuda.is_available()
    weight = build_weight(vocab, corpus)
    if cuda_flag:
        weight = weight.cuda()
    model = build_model(vocab, weight)    #建立模型
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        logger.info("epoch %d begin" % epoch)
        model.train()
        watch_loss = []
        train_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            if cuda_flag:
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            train_loss.append(float(loss))
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
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
