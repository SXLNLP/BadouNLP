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

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.lstm(x)
        # 使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()
        self.triplet_margin = config["triplet_margin"]

    # 计算余弦距离 1-cos(a,b)
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        margin = self.triplet_margin if margin is None else margin
        diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)])  # greater than

    def forward(self, sentence1, sentence2=None, sentence3=None, target=None):
        # 传入三个句子(anchor, positive, negative)
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.sentence_encoder(sentence1)  # anchor
            vector2 = self.sentence_encoder(sentence2)  # positive
            vector3 = self.sentence_encoder(sentence3)  # negative
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        # 传入两个句子(保持原来的功能)
        elif sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            else:
                return self.cosine_distance(vector1, vector2)
        # 单独传入一个句子
        else:
            return self.sentence_encoder(sentence1)


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
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    s3 = torch.LongTensor([[4,3,2,1], [1,1,1,1]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, s3)  # 测试三元组输入
    print(y)
    y = model(s1, s2, None, l)  # 测试二元组输入
    print(y)
