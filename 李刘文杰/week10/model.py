# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        hidden_size= config["hidden_size"]
        vocab_size = config["vocab_size"]
        #self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0) 
        #双向LSTM self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert=BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size,vocab_size)#全连接层
        

        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):

        if y is not None:#训练模式
                #使用mask机制，即构建倒三角矩阵，使[上文无法与下文产生关联
             mask=torch.tril(torch.ones((x.shape[0], x.shape[1],x.shape[1])))
             #print(mask,mask.shape)
             if torch.cuda.is_available():
                 mask=mask.cuda()
             x,_=self.bert(x, attention_mask=mask)
             y_pred = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
             return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1)) # 返回每个标签的概率分布
        
        else:#预测模式
             x,_= self.bert(x)
             y_pred = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
             return torch.softmax(y_pred, dim=-1) # 返回每个标签的概率分布


if __name__ == "__main__":
    from config import Config
    model = LanguageModel(Config)
