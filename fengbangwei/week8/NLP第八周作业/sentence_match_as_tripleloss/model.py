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
        # 初始化Cosine Embedding Loss函数
        # 该损失函数用于衡量两个向量间的余弦相似度，并根据它们是否应当相似返回损失
        # 主要用于对比学习或度量学习任务中，以学习向量表示
        # self.loss = nn.CosineEmbeddingLoss()

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
        return torch.mean(diff[diff.gt(0)])  # greater than

    # sentence : (batch_size, max_length)
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor = None,
                negative: torch.Tensor = None,
                margin: float = None) -> torch.Tensor:
        """
        前向传播
        Args:
            anchor: 锚点样本 [batch_size, seq_len]
            positive: 正样本 [batch_size, seq_len] (可选)
            negative: 负样本 [batch_size, seq_len] (可选)
            margin: triplet loss的间隔参数
        Returns:
            当提供positive和negative时返回loss
            否则返回anchor的编码向量
        """
        # 类型检查
        assert anchor.dtype == torch.long, "输入必须是LongTensor"

        if positive is not None and negative is not None:
            # 三元组模式
            anchor_vec = self.sentence_encoder(anchor)
            positive_vec = self.sentence_encoder(positive)
            negative_vec = self.sentence_encoder(negative)
            return self.cosine_triplet_loss(anchor_vec, positive_vec, negative_vec, margin)
        else:
            # 单句编码模式
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
    # s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0], [3, 2, 3, 4]])
    # s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    # l = torch.LongTensor([[1], [0]])
    s1 = torch.LongTensor([[1, 2, 3, 0]])
    s2 = torch.LongTensor([[2, 2, 0, 0]])
    s3 = torch.LongTensor([[3, 2, 3, 4]])
    y = model(s1, s2, s3)
    print(y)
    # print(model.state_dict())
