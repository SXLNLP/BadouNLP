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
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)    # padding_idx=0: 表示索引为 0 的 token 是 padding，不参与更新。
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
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.1, p=2)  # 添加三元组损失函数

    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def forward(self, sentence1, sentence2=None, target=None, sentence3=None):
        if sentence2 is not None and sentence3 is not None:  # 使用三元组损失函数
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            vector3 = self.sentence_encoder(sentence3)
            return self.triplet_loss(vector1, vector2, vector3)
        elif sentence2 is not None:  # 使用余弦损失函数
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            return self.cosine_loss(vector1, vector2, target.squeeze())
        else:  # 单独传入一个句子时，返回向量
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
