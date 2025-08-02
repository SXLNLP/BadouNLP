#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel,BertTokenizer
import re
from config import Config

"""
基于pytorch的BERT模型
"""


class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        hidden_size = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size
        self.cls = nn.Linear(hidden_size, vocab_size)  # 输出词表概率

        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """动态生成因果掩码（下三角矩阵，0=可见，1=不可见）"""
        # 形状：(1, 1, seq_len, seq_len) （兼容多头注意力的广播机制）
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # 扩展为 (batch_size=1, num_heads=1, ...)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None,attention_mask=None, causal_mask=None):
        # 如果没有提供因果掩码，默认不使用（仅处理填充）
        if causal_mask is not None:
            # 合并因果掩码和原始注意力掩码（因果掩码优先级更高）
            # 因果掩码形状：(batch_size, 1, seq_len, seq_len)
            # attention_mask形状：(batch_size, 1, 1, seq_len)
            # 最终掩码：因果掩码（0=可见，1=不可见）与 attention_mask（0=可见，1=不可见）的逻辑或
            # 即：如果因果掩码或原始掩码标记为不可见，则最终不可见
            combined_mask = causal_mask.logical_or(attention_mask)
        else:
            combined_mask = attention_mask

        # BERT前向传播，传入合并后的注意力掩码 ,获取最后一层隐藏状态
        outputs = self.bert(
            input_ids=x,
            attention_mask=combined_mask,  # 使用合并后的掩码
            return_dict=False  # 返回元组（兼容旧版本）# 返回元组 (last_hidden_state, pooler_output)
        )
        hidden_states = outputs[0]  # last_hidden_state 的形状：(batch_size, seq_len, hidden_size)

        # 计算词表概率（仅预测目标部分的token）
        y_pred = self.cls(hidden_states)  # (batch_size, seq_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表 bert就不需要了
def build_vocab(vocab_path,use_bert=0):
    if use_bert == 0:
        vocab = {"<pad>":0}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                char = line[:-1]       #去掉结尾换行符
                vocab[char] = index + 1 #留出0位给pad token
    else:
        # 加载中文 BERT 分词器（自动下载词表）
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        vocab = tokenizer.vocab  # 词表字典：{token: id}
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
def build_sample(vocab,window_size, corpus):
    #print(window_size)
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    #print("window-----",window, "target------",target)
    #window----- 目前的形势，他们没有 target------ 前的形势，他们没有赢

    x = [vocab.get(word, vocab["[UNK]"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["[UNK]"]) for word in target]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab,window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(config):
    model = LanguageModel(config)
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


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    #char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    vocab = build_vocab("vocab.txt",use_bert=1)       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(Config)    #建立模型
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
