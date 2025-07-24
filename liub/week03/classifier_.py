# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/5/28
# @Author      : liuboyuan

# 构造随机包含a的字符串, 使用rnn进行多分类, 
# 类别为a第一次出现在字符串的位置。

import torch
import torch.nn as nn

class TorchModel(nn.Module):
    def __init__(self, vector_dim, num_classes, vocab, hidden_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(input_size=vector_dim,
                          hidden_size=hidden_dim,
                          bias=False,
                          batch_first=True)
        self.classify = nn.Linear(hidden_dim, num_classes)     # 线性层，输出 num_classes
        self.loss = nn.CrossEntropyLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x_emb = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # output: (batch_size, sen_len, hidden_dim)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        output, h_n = self.rnn(x_emb)

        # 取最后一个时间步的隐藏状态
        # h_n 的形状是 (1, batch_size, hidden_dim) 因为 num_layers=1, bidirectional=False
        last_hidden_state = h_n.squeeze(0) # -> (batch_size, hidden_dim)

        logits = self.classify(last_hidden_state)  # -> (batch_size, num_classes)

        if y is not None:
            # y 已经是 LongTensor 并且形状是 (batch_size)
            return self.loss(logits, y)   #预测值和真实值计算损失
        else:
            return torch.softmax(logits, dim=-1)                 #输出预测结果