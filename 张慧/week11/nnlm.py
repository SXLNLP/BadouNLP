#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertTokenizer, BertModel
import json

"""
基于pytorch的LSTM语言模型
"""

config = {
    "pretrain_model_path": r"../bert-base-chinese",  # 或本地路径
    "max_length": 512,  # BERT最大长度限制
    "hidden_size": 768,  # BERT隐藏层大小
    "num_layers": 1,  # 层数
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": True,
    "class_num": 9,
    "vocab_path": "../bert-base-chinese/vocab.txt",  # 或本地路径
}

class LanguageModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.classify = nn.Linear(self.bert.config.hidden_size, vocab_size)

        # 修改损失函数，添加ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略-1的标签

    def forward(self, x,len_s1, y=None):
        mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
        mask[:, :len_s1, :len_s1] = 1
        # print("mask shape:", mask.shape)

        if torch.cuda.is_available():
            mask = mask.cuda()
        # 使用BERT进行编码
        sequence_output, _ = self.bert(x, attention_mask=mask)
        # print("sequence_output shape:", sequence_output.shape)
        # sequence_output shape: (batch_size, seq_length, hidden_size)

        # 使用BERT的输出进行分类
        y_pred = self.classify(sequence_output)

        if y is not None:
            y_padding = torch.full((x.shape[0], len_s1), -1,
                                   dtype=torch.int64)  # 创建一个填充的tensor，形状为(batch_size, len_s1)，填充值为-1
            if torch.cuda.is_available():
                y_padding = y_padding.cuda()
            y = torch.cat([y_padding, y], dim=1)  # 在第1维(行方向)拼接
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  #去掉结尾换行符
            vocab[char] = index + 1  #留出0位给pad token
    return vocab


#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def read_json(file_path):
    """
    读取JSON文件并返回数据
    :param file_path: JSON文件路径
    :return: JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def build_sample(corpus, tokenizer, window_size):
    title = corpus.get('title')  # s1
    content = corpus.get('content')  # s2

    # 分别编码 title 和 content
    title_enc = tokenizer(title,
                          add_special_tokens=False,
                          padding='max_length',
                          truncation=True,
                          max_length=window_size,
                          return_tensors="pt")

    content_enc = tokenizer(content,
                            add_special_tokens=False,
                            padding='max_length',
                            truncation=True,
                            max_length=window_size,
                            return_tensors="pt")

    # 拼接 input_ids: [s1, s2]
    input_ids = torch.cat([title_enc.input_ids, content_enc.input_ids], dim=1).squeeze(0)  # [1000]
    len_s1 = title_enc.input_ids.shape[1]  # e.g., 500

    # target 只对 content 做监督（即 s2）
    targets = content_enc.input_ids.squeeze(0)  # [500]

    return input_ids, targets, len_s1


def build_dataset_from_corpus(sample_length, tokenizer, window_size, corpus):
    dataset_x, dataset_y = [], []
    for i in range(sample_length):
        item = corpus[i]
        x, y, len_s1 = build_sample(item, tokenizer, window_size)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.stack(dataset_y), len_s1


#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x, dataset_y = [], []
    for _ in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.stack(dataset_y)


#建立模型
def build_model(config, vocab_size):
    model = LanguageModel(config, vocab_size)
    return model


#文本生成测试代码
def generate_sentence(openings, model, tokenizer,window_size):
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
            y = model(x,len_s1=window_size)[0][-1]
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
            pred_prob_distribute = model(x,len_s1=100)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20  #训练轮数
    batch_size = 64  #每次训练样本个数
    train_sample = 50000  #每轮训练总共训练的样本总数
    char_dim = 256  #每个字的维度
    window_size = 30  #样本文本长度
    vocab_size = 21128

    tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
    corpus = read_json(corpus_path)  #加载JSON语料
    model = build_model(config, vocab_size)  #建立模型

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  #建立优化器
    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            x, y, len_s1 = build_dataset_from_corpus(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  #梯度归零
            loss = model(x,len_s1, y)  #计算loss
            loss.backward()  #计算梯度
            optim.step()  #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换：", model, tokenizer,window_size))
        print(generate_sentence("国际通用航空大会沈阳飞行家表演队一飞机发生坠机，伤亡不明：", model, tokenizer,window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
