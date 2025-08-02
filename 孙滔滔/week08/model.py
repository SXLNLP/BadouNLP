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


class TripletNetwork(nn.Module):
    def __init__(self, config):
        super(TripletNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) #greater than

    #sentence : (batch_size, max_length)
    def forward(self, anchor=None, positive=None, negative=None, sentence1=None, sentence2=None):
        # Triplet Loss训练:接收anchor, positive, negative
        if anchor is not None and positive is not None and negative is not None:
            anchor_vec = self.sentence_encoder(anchor)  # (batch_size, hidden_size)
            positive_vec = self.sentence_encoder(positive)  # (batch_size, hidden_size)
            negative_vec = self.sentence_encoder(negative)  # (batch_size, hidden_size)

            # 计算triplet loss
            loss = self.loss(anchor_vec, positive_vec, negative_vec)
            return loss

        # 原有的双句子输入
        elif sentence1 is not None and sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            return self.cosine_distance(vector1, vector2)

        # 单独传入一个句子时，进行向量化
        elif anchor is not None:
            return self.sentence_encoder(anchor)
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
    model = TripletNetwork(Config)
    anchor = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    positive = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    negative = torch.LongTensor([[4, 3, 2, 1], [1, 1, 1, 1]])
    loss = model(anchor=anchor, positive=positive, negative=negative)
    print(f"Triplet Loss: {loss}")