"""
在预训练的Bert模型上使用使用新闻数据尝试实现sft训练
"""
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

class SFTModel(nn.Module):
    def __init__(self, hidden_size,vocab_size,pretrain_model_path):
        super(SFTModel, self).__init__()
        self.bert= BertModel.from_pretrained(pretrain_model_path,return_dict=False,attn_implementation='eager')

        self.classify=nn.Linear(hidden_size, vocab_size)
        
        self.loss=nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充的标签

    def forward(self,x,y=None):
        if y is not None:
            #训练模式,构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _= self.bert(x, attention_mask=mask)
            y_pred=self.classify(x) #输出形状：[batch_size,vocab_size]
            loss=self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
            return loss
        else:
            #预测时，不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # 输出形状：[batch_size,vocab_size]
            return torch.softmax(y_pred, dim=-1)  # 返回概率分布
        
#加载语料,加载语料, 用title当成假想的prompt，content当成假想的answer
def load_corpus(path):
    corpus=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus



#构造掩码
def create_mask(s1,s2):
    len_s1=s1+2 #加上[CLS]和[SEP]
    len_s2=s2+2 #加上[CLS]和[SEP]
    #创建掩码
    mask=torch.ones((len_s1+len_s2,len_s1+len_s2))
    #遍历s1的每个token
    for i in range(len_s1):
        #s1的当前的token不能看到s2的任何token
        mask[i,len_s1:]=0
    #遍历s2的每个token
    for i in range(len_s2):
        #s2的当前的token不能看到s2后面的任何token
        mask[len_s1+i,len_s1+i+1:]=0
    return mask
#sft的数据构造
#loss只计算答案按部分，通过mask矩阵，让上下文之间没有交互
#label中使用-1，表示不参与训练
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        #构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = create_mask(len(prompt_encode), len(answer_encode))
        #padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

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

#建立模型
def build_model(char_dim,vocab_size,pretrain_model_path):
    model=SFTModel(char_dim,vocab_size,pretrain_model_path)
    return model    

#训练模型
def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 32      #每次训练样本个数
    char_dim = 768        #每个字的维度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    max_length = 50       #样本文本长度

    pretrain_model_path = r"D:/Code/py/八斗nlp/20250622/week6 语言模型和预训练/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  #建立数据集
    model = build_model(char_dim,vocab_size,pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        print(f"第{epoch + 1}轮训练开始")
        model.train()
        watch_loss=[]
        #计算每个batch的样本数
        for x,mask,y in train_data:#构建一组训练样本
             
            if torch.cuda.is_available():
                x,mask,y = x.cuda(),mask.cuda(), y.cuda()
            #梯度更新
            optim.zero_grad()
            #计算loss
            loss = model(x, y)
            #反向传播
            loss.backward()
            #更新参数
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"D:/Code/py/八斗nlp/20250720/第十周/week10 文本生成问题/bert语言模型生成文本/vocab.txt", False)
