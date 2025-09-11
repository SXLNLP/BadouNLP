# bert+mask 训练自生成式回归任务
import torch
import math
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
import torch.nn as nn
import json
import re
from torch.optim import Adam, SGD

bert_path="/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese"
bert_tokenizer=BertTokenizer.from_pretrained(bert_path)
bert = BertModel.from_pretrained(bert_path, return_dict=False)
text_path="lstm语言模型生成文本/corpus.txt"


seq_dataset=[]   # 54967
batch_size=10    # batch_size 是10 那一个batch训练10个句子，10个句子对x_padded,y_padded, 用bert cls 和sep 代替bos和eos  
epoch=10
batch_num=len(seq_dataset)//batch_size//epoch   # 5496

#句子数据，生成器每个batch
def batch_dataset_generator():
    for i in range(0,len(seq_dataset),batch_size):
        yield seq_dataset[i:i+batch_size]
        

# 使用bert 作batch encoder,用batch内的最大句子长度，而不用全局最大句子长度，可以节省开销
def batch_dataset_encoder(batch):
    max_length=max([len(line) for line in batch])
    inputs=bert_tokenizer(batch,return_tensors='pt', padding=True,max_length=max_length,is_split_into_words=True)
    embedded_inputs=bert(**inputs)
    return embedded_inputs

class Autoregressive(nn.modules):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        """
        一个基于 TransformerDecoder 的简单自回归模型。

        :param vocab_size: 词汇表大小
        :param d_model: 模型的隐藏维度 (embedding dimension)
        :param nhead: 多头注意力的头数
        :param num_decoder_layers: 解码器层的数量
        :param dim_feedforward: 前馈网络的维度
        """
        super(Autoregressive, self).__init__()
        self.loss=nn.CrossEntropyLoss() 
        self.embedding=nn.Embedding(vocab_size, hidden_size, padding_idx=0)
    def forward():



    # inputs=bert_tokenizer(text,return_tensors='pt', padding=True,max_length=241,is_split_into_words=True)#送入的是字符列表
    # embedded_inputs=bert(**inputs)
    # print(inputs)


if __name__=="__main__":
    with open(text_path, "r", encoding="gbk") as t:
        for line in t:  # 文章已经分好句子，一个line是一个句子          
          if line.strip():
            seq_dataset.append(line.strip())
    # 实例化生成器    
    batch_generator=batch_dataset_generator()
    for i in range(epoch):
        for j in range(batch_num):
            batch=next(batch_generator)
            embedded_inputs=batch_dataset_encoder(batch)
            

    
