# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

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

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        # 这里不再使用 CosineEmbeddingLoss，因为我们要使用三元组损失
        # self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=0.5):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)])  # greater than

    #sentence : (batch_size, max_length)
    def forward(self, anchor, positive=None, negative=None):
        # 同时传入三个句子（锚点、正样本、负样本）
        if negative is not None and positive is not None:
            anchor_vector = self.sentence_encoder(anchor)  # vec:(batch_size, hidden_size)
            positive_vector = self.sentence_encoder(positive)
            negative_vector = self.sentence_encoder(negative)
            # 计算三元组损失
            return self.cosine_triplet_loss(anchor_vector, positive_vector, negative_vector)
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(anchor)


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
    model = SiameseNetwork(Config)
    a = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    p = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    n = torch.LongTensor([[4,3,2,1], [1,3,2,4]])
    y = model(a, p, n)
    print(y)
    # print(model.state_dict())