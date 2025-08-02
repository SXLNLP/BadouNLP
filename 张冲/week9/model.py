# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x, _ = self.encoder(x)  # input shape:(batch_size, sen_len) -> (batch_size, sen_len, hidden_size)
        predict = self.classify(x)  # ouput:(batch_size, sen_len, hidden_size) -> (batch_size * sen_len, num_tags)
        if target is not None:
            # (number, class_num), (number)
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
             return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    model = TorchModel(Config)
