#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer,BertConfig,BertModel

"""
使用BERT模型进行文本生成
"""
#设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#BERT模型配置
BERT_CONFIG = BertConfig.from_pretrained("bert-base-chinese")
BERT_CONFIG.is_decoder = True #设置为解码器（自带掩码）
BERT_CONFIG.add_cross_attention = False #无交叉注意力
BERT_CONFIG.return_dict = False
#自回归的BERT模型
BERT_MODEL = BertModel.from_pretrained("bert-base-chinese", config=BERT_CONFIG)
#BERT词表
BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-chinese")
VOCAB_SIZE = len(BERT_TOKENIZER)

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        #self.embedding = nn.Embedding(len(vocab), input_dim)
        self.layer = BERT_MODEL
        hidden_size = BERT_CONFIG.hidden_size
        self.classify = nn.Linear(hidden_size, VOCAB_SIZE)
        #self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x,xm=None, y=None):
        x = self.layer(x,attention_mask=xm)[0]
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
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
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus):
    start = random.randint(0, len(corpus) - window_size - 1)
    window = corpus[start : start + window_size]
    target = corpus[start + 1 : start + 1 + window_size]

    # 下面每次都强制输出固定长度 window_size 的 id 列表
    enc_x = BERT_TOKENIZER(
        window,
        add_special_tokens=False,
        max_length=window_size,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc_y = BERT_TOKENIZER(
        target,
        add_special_tokens=False,
        max_length=window_size,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # squeeze 是因为 return_tensors="pt" 会多加一个 batch 维
    x = enc_x["input_ids"].squeeze(0)
    x_mask = enc_x["attention_mask"].squeeze(0) # 获取 attention_mask
    y = enc_y["input_ids"].squeeze(0)
    return x.tolist(), x_mask.tolist() ,y.tolist()

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    dataset_xm = []
    for i in range(sample_length):
        x, xmask, y = build_sample(window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_xm.append(xmask)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_xm), torch.LongTensor(dataset_y)

#建立模型
def build_model():
    model = LanguageModel().to(DEVICE)
    return model

#文本生成测试代码
def generate_sentence(openings, model, window_size):
    #reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            enc_x = BERT_TOKENIZER(
            openings[-window_size:],
            add_special_tokens=False,
            max_length=window_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            )
            x = enc_x["input_ids"].squeeze(0).tolist()
            x_mask = enc_x["attention_mask"].squeeze(0).tolist()  # 获取 mask

            #x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            x = x.to(DEVICE)
            x_mask_tensor = torch.LongTensor([x_mask]).to(DEVICE)  # 转为 Tensor

            y = model(x,x_mask_tensor)[0][-1]
            index = sampling_strategy(y)
            # 将index转换为字符串
            pred_char = BERT_TOKENIZER.decode([index], skip_special_tokens=True)            
            #pred_char = reverse_vocab[index]
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
# def calc_perplexity(sentence, model, vocab, window_size):
#     prob = 0
#     model.eval()
#     with torch.no_grad():
#         for i in range(1, len(sentence)):
#             start = max(0, i - window_size)
#             window = sentence[start:i]
#             x = [vocab.get(char, vocab["<UNK>"]) for char in window]
#             x = torch.LongTensor([x])
#             target = sentence[i]
#             target_index = vocab.get(target, vocab["<UNK>"])
#             x = x.to(DEVICE)
#             pred_prob_distribute = model(x)[0][-1]
#             target_prob = pred_prob_distribute[target_index]
#             prob += math.log(target_prob, 10)
#     return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    #char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    #vocab = build_vocab("homework/vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    #print(corpus)
    model = build_model()    #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x,xm, y = build_dataset(batch_size, window_size, corpus) #构建一组训练样本
            x, xm, y = x.to(DEVICE),xm.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()    #梯度归零
            loss = model(x,xm, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能", model,  window_size))        
        print(generate_sentence("李慕站在山路上，深深地呼吸", model,  window_size))
        print(generate_sentence("在半年之前", model,  window_size))
        print(generate_sentence("李慕站在山路上，深", model,  window_size))
        print(generate_sentence("李慕拔出", model,  window_size))
        print(generate_sentence("哈哈哈哈", model,  window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("homework/corpus.txt", False)
