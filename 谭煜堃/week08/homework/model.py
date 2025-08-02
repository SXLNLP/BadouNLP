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
        vocab_size = config["vocab_size"] + 1 # vocab_size就是词表大小，0是padding的索引，因此实际的词表大小要比词总数多1
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
        self.loss = self.cosine_triplet_loss

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
    def forward(self, sentence_a, sentence_p=None, sentence_n=None, margin=None):
        #同时传入两个句子
        if (sentence_p is not None) and (sentence_n is not None):
            vector_a = self.sentence_encoder(sentence_a) #vec:(batch_size, hidden_size)
            vector_p = self.sentence_encoder(sentence_p)
            vector_n = self.sentence_encoder(sentence_n)
            loss = self.loss(vector_a, vector_p, vector_n, margin)
            return loss
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence_a)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import GlobalConfigManager
    config_manager = GlobalConfigManager()
    Config = config_manager.get_config()
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    Config["epoch_data_size"] = 200
    Config["positive_sample_rate"] = 0.5
    model = SiameseNetwork(Config)
    sp = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    sn = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    sa = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    loss_example = model(sa, sp, sn, margin=torch.tensor(0.1))
    print(loss_example)
    encoded_example = model(sa)
    print(encoded_example)
    # print(model.state_dict())