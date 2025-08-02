# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络模型结构
"""


class BertGeneratorModel(nn.Module):
    def __init__(self, config):
        super(BertGeneratorModel, self).__init__()

        self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        self.vocab_size = config["vocab_size"]

        self.classify = nn.Linear(hidden_size, self.vocab_size)
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失
        self.pooling = nn.AvgPool1d(self.vocab_size)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = torch.from_numpy(np.array(x)).squeeze(dim=-1)
        mask = torch.masked_fill(torch.triu(torch.ones_like(x)), x == 0, 0)
        x = self.encoder(x, attention_mask=mask)[0]
        x = self.pooling(torch.transpose(x, -1, -2)).squeeze(-1)

        predict = self.classify(x[0])  # output shape: probability(batch_size, vocab_size)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))
