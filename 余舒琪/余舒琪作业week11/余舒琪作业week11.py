#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json

from openpyxl.styles.builtins import output
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

"""
使用bert添加mask实现sft训练语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size, bert_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False, is_decoder=True)
        self.classify = nn.Linear(input_dim, vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, attention_mask=None):
        if attention_mask is not None:
            attention_mask = (1 - attention_mask) * -1e9
        x = self.bert(x, attention_mask=attention_mask)[0]
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=-100)
        else:
            return torch.softmax(y_pred, dim=-1)

class DataGenerator:
    def __init__(self, data_path, tokenizer, max_length):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load()

    def load(self):
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                question = line["title"]
                answer = line["content"]
                self.prepare_data(question, answer)
        return

    def prepare_data(self, question, answer):
        question_id = self.tokenizer.encode(question, add_special_tokens=False)
        answer_id = self.tokenizer.encode(answer, add_special_tokens=False)
        cls_id = self.tokenizer.cls_token_id #以[CLS]分割question和answer
        sep_id = self.tokenizer.sep_token_id #以[SEP]作为answer结尾
        input_ids = question_id + [cls_id] + answer_id #question[CLS]answer
        input_ids = self.padding(input_ids)
        output_ids = [-100] * len(question) + answer_id + [sep_id] #[-100,...]answer[SEP]
        output_ids = self.padding(output_ids)
        assert len(input_ids) == len(output_ids)
        mask1 = torch.ones(len(input_ids), len(question) + 1)
        mask2 = torch.zeros(len(question) + 1, len(input_ids) - (len(question) + 1))
        mask3 = torch.tril(torch.ones(len(input_ids) - (len(question) + 1), len(input_ids) - (len(question) + 1)))
        mask = torch.cat((mask2, mask3), dim=0)
        mask = torch.cat((mask1, mask), dim=1)
        self.data.append([torch.LongTensor(input_ids), torch.LongTensor(output_ids), torch.FloatTensor(mask)])
        return

    def padding(self, vector):
        pad_id = self.tokenizer.pad_token_id
        vector = vector[:self.max_length]
        vector += [pad_id] * (self.max_length - len(vector))
        return vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path, tokenizer, batch_size, max_length, shuffle=True):
    dg = DataGenerator(data_path, tokenizer, max_length)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl

#建立模型
def build_model(char_dim, vocab_size, bert_path):
    model = LanguageModel(char_dim, vocab_size, bert_path)
    return model

#文本生成测试代码
def generate_sentence(question, model, tokenizer):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    model.eval()
    with torch.no_grad():
        answer = ""
        input_ids = tokenizer.encode(question, add_special_tokens=False)
        index = cls_id
        #生成了[SEP]，或生成文本超过110字则终止迭代
        while index != sep_id and len(answer) <= 110:
            input_ids.append(index)
            x = torch.LongTensor([input_ids])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1] #取最后一个字输出的概率分布
            index = sampling_strategy(y)
            pred_char = "".join(tokenizer.decode(index))
            answer += pred_char
    return answer

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        # return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
        return np.random.choice(len(prob_distribution), p=prob_distribution)

def train(data_path, bert_path, save_weight=True):
    epoch_num = 30        #训练轮数
    batch_size = 32       #每次训练样本个数
    char_dim = 768        #每个字的维度
    max_length = 130      #样本文本长度
    vocab_size = 21128    #字表大小
    learning_rate = 0.001 #学习率
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_data = load_data(data_path, tokenizer, batch_size, max_length, shuffle=True)
    model = build_model(char_dim, vocab_size, bert_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            x, y, mask = batch_data
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("广州诺如病毒", model, tokenizer))
        print(generate_sentence("怎样选面膜", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(data_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json",r"D:\学习\ai\week6\预习\bert-base-chinese", False)
