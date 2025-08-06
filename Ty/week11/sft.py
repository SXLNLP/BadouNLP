#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
from transformers import BertModel,BertTokenizer
from torch.utils.data import Dataset, DataLoader,TensorDataset
import json

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\往期\bert-base-chinese", return_dict=False)
        self.bert.config.is_decoder=True
        input_dim = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size
        #print(input_dim, vocab_size)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index = -1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y = None, past_kv = None):
        if y is not None:
            seq_len = x.shape[1]
            pad_id = self.bert.config.pad_token_id
            ignore_id = self.loss.ignore_index
            mask = (torch.tril(torch.ones(seq_len,seq_len,device='cuda')).bool().unsqueeze(0)
                    | (y == ignore_id).unsqueeze(-2)) & (x != pad_id).unsqueeze(-2)
            x = self.bert(x, attention_mask = mask)[0]
            y_pred = self.classify(x)   #output shape:(batch_size, seq_len, vocab_size)
            return self.loss(y_pred.transpose(-2,-1), y)
        else:
            x, _, past_kv = self.bert(x, past_key_values=past_kv, use_cache=True)
            #x = self.bert(x)[0]
            print(len(past_kv), past_kv[0].shape)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1), past_kv

#构建一个样本
def build_sample(title, content, max_len, tokenizer):
    title_ids = tokenizer.encode(title, add_special_tokens=False)   #将字转换成序号
    content_ids = tokenizer.encode(content, add_special_tokens=False)
    x = title_ids + [tokenizer.cls_token_id] + content_ids
    y = [-1]*len(title_ids) + content_ids + [tokenizer.sep_token_id]
    x = padding(x, max_len, tokenizer.pad_token_id)
    y = padding(y, max_len, -1)
    return [torch.LongTensor(x), torch.LongTensor(y)]

def padding(input_id, length, pad_id):
    input_id = input_id[:length]
    input_id += [pad_id] * (length - len(input_id))
    return input_id

#建立数据集
def build_dataset(max_length, path, tokenizer, batch_size):
    dataset = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            dataset.append(build_sample(title, content, max_length, tokenizer))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

#建立模型
def build_model():
    model = LanguageModel()
    return model

#文本生成测试代码
def generate_sentence_(openings, model, tokenizer):
    #使用kv缓存
    model.eval()
    with torch.no_grad():
        x = tokenizer.encode(openings, add_special_tokens=False, return_tensors='pt').cuda()
        y, past_kv = model(x)
        index = tokenizer.cls_token_id
        output = ""
        while index != tokenizer.sep_token_id and len(openings) <= 150:
            y, past_kv = model(torch.tensor([[index]],device='cuda'), past_kv=past_kv)
            index = sampling_strategy(y[0][-1])
            output += tokenizer.decode(index)
    return output

def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        x = tokenizer.encode(openings+tokenizer.cls_token, add_special_tokens=False, return_tensors='pt').cuda()
        index = -1
        while index != tokenizer.sep_token_id and x.shape[1] <= 150:
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            x = torch.cat((x, torch.tensor([[index]], device='cuda')), 1)
    return tokenizer.decode(x[0])

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
    epoch_num = 60     #训练轮数
    batch_size = 64       #每次训练样本个数
    tokenizer = BertTokenizer.from_pretrained(r"D:\BaiduNetdiskDownload\往期\bert-base-chinese")
    dataset = build_dataset(150, corpus_path, tokenizer, batch_size)  # 加载语料
    model = build_model()  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
        print("gpu可以使用，迁移模型至gpu")
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in dataset:
            x, y = batch #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence_("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer))
        print(generate_sentence_("卫生计生委国际司司长：真正的免费医疗不存在", model, tokenizer))
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
