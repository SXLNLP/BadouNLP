# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

"""
建立网络模型结构
"""


class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, hidden_size)

    def forward(self, x):
        if x.dim() == 1:
            # 添加 batch 维度 [seq_len] -> [1, seq_len]
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Expected input to be 1D or 2D tensor, got {x.dim()}D")

        # 检查输入是否是 LongTensor 类型（用于索引）
        if x.dtype != torch.long:
            raise TypeError(f"Expected input to be LongTensor (dtype=torch.long), got {x.dtype}")

        # 嵌入层
        x = self.embedding(x)
        # print("嵌入后 x 的形状:", x.shape)

        # 确认维度
        if x.dim() != 3:
            raise ValueError(f"Expected input to be 3D tensor after embedding, got {x.dim()}D")

        # 如果是二维输入 [batch_size, seq_len]，则扩展为 [batch_size, seq_len, embed_size]
        # 但 embedding 已经输出三维张量，所以下面这步不是必须的

        # 线性变换到 hidden_size
        batch_size, seq_len, embed_size = x.shape
        x = x.view(-1, embed_size)  # [batch_size * seq_len, embed_size]
        x = self.fc(x)  # [batch_size * seq_len, hidden_size]
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        # print("经过 fc 后 x 的形状:", x.shape)

        # 对序列维度做 max pooling
        # x 的形状应该是 [batch_size, seq_len, hidden_size]
        # 转置成 [batch_size, hidden_size, seq_len] 以使用 max_pool1d
        x = x.transpose(1, 2)  # -> [batch_size, hidden_size, seq_len]
        # print("转置后 x 的形状:", x.shape)

        # 使用 max_pool1d 做全局池化（kernel_size = seq_len）
        x = F.max_pool1d(x, kernel_size=x.shape[2])  # [batch_size, hidden_size, 1]

        # squeeze 掉最后的 1 维
        x = x.squeeze()  # [batch_size, hidden_size]
        # print("池化并 squeeze 后 x 的形状:", x.shape)

        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(
            vocab_size=config["vocab_size"],
            embed_size=config.get("embed_size", 100),  # 默认值也可以设为 100
            hidden_size=config["hidden_size"]
        )

    def forward(self, anchor, positive, negative):
        a_vec = self.sentence_encoder(anchor)
        p_vec = self.sentence_encoder(positive)
        n_vec = self.sentence_encoder(negative)
        return a_vec, p_vec, n_vec


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    Config["hidden_size"] = 16
    model = SiameseNetwork(Config)

    a = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    p = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    n = torch.LongTensor([[4,5,6,7], [5,5,0,0]])

    a_vec, p_vec, n_vec = model(a, p, n)
    print("Anchor Vector Shape:", a_vec.shape)
    print("Positive Vector Shape:", p_vec.shape)
    print("Negative Vector Shape:", n_vec.shape)
