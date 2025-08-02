import torch
import torch.nn as nn
from torch.optim import Adam, SGD

"""
网络模型（支持三元组损失）
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, max_len, hidden_size)
        x = self.layer(x)      # 线性变换
        x = self.dropout(x)
        # 最大池化获取句向量
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.margin = config["margin"]  # 三元组损失的margin

    def cosine_distance(self, tensor1, tensor2):
        """计算余弦距离 (1 - cosine相似度)"""
        tensor1 = nn.functional.normalize(tensor1, dim=-1)
        tensor2 = nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
        return 1 - cosine

    def triplet_loss(self, anchor, positive, negative):
        """三元组损失: max(0, d(a,p) - d(a,n) + margin)"""
        d_ap = self.cosine_distance(anchor, positive)
        d_an = self.cosine_distance(anchor, negative)
        loss = torch.mean(torch.max(d_ap - d_an + self.margin, torch.zeros_like(d_ap)))
        return loss

    def forward(self, anchor, positive=None, negative=None):
        # 三元组输入（训练阶段）
        if positive is not None and negative is not None:
            anchor_vec = self.sentence_encoder(anchor)
            positive_vec = self.sentence_encoder(positive)
            negative_vec = self.sentence_encoder(negative)
            return self.triplet_loss(anchor_vec, positive_vec, negative_vec)
        # 单句输入（编码阶段）
        else:
            return self.sentence_encoder(anchor)


def choose_optimizer(config, model):
    if config["optimizer"] == "adam":
        return Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        return SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"不支持的优化器: {config['optimizer']}")