# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.margin = config.get("margin", 0.1)  # 添加margin参数

    def cosine_distance(self, tensor1, tensor2):
        """计算余弦距离 1-cos(a,b)"""
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, anchor, positive, negative):
        """三元组损失函数"""
        ap_distance = self.cosine_distance(anchor, positive)
        an_distance = self.cosine_distance(anchor, negative)
        losses = torch.relu(ap_distance - an_distance + self.margin)
        return torch.mean(losses)

    def forward(self, anchor, positive=None, negative=None, target=None):
        """
        输入模式:
        1. 训练模式: 提供anchor, positive, negative三个句子
        2. 向量化模式: 只提供anchor句子
        """
        anchor_vec = self.sentence_encoder(anchor)
        
        # 训练模式
        if positive is not None and negative is not None:
            positive_vec = self.sentence_encoder(positive)
            negative_vec = self.sentence_encoder(negative)
            return self.cosine_triplet_loss(anchor_vec, positive_vec, negative_vec)
        # 向量化模式
        else:
            return anchor_vec


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    Config["margin"] = 0.1  # 添加margin配置
    
    model = SiameseNetwork(Config)
    
    # 测试三元组输入
    anchor = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    positive = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    negative = torch.LongTensor([[4,3,2,1], [1,1,1,1]])
    
    loss = model(anchor, positive, negative)
    print("Triplet Loss:", loss)
    
    # 测试向量化
    vec = model(anchor)
    print("Vector shape:", vec.shape)
