# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        model_type = config["model_type"]

        self.use_bert = False
        if model_type == "bert":
            if model_type == "bert":
                try:
                    # 先尝试从本地加载
                    self.encoder = BertModel.from_pretrained(
                        config["local_model_dir"] + "/bert-base-chinese"
                    )
                except:
                    # 本地不存在则从官网下载
                    self.encoder = BertModel.from_pretrained(
                        config["pretrain_model_path"],
                        cache_dir=config["local_model_dir"]
                    )
        else:
            vocab_size = config["vocab_size"]
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

            if model_type == "lstm":
                self.embedding = nn.Embedding(
                    config["vocab_size"],
                    hidden_size,
                    padding_idx=0
                )
                self.encoder = nn.LSTM(
                    hidden_size,  # 修改：输入维度与embedding维度一致
                    hidden_size // 2,  # 双向LSTM需要减半
                    num_layers=config["num_layers"],
                    batch_first=True,
                    bidirectional=True
                )
                self.hidden_size = hidden_size  # 保存最终输出维度
            elif model_type == "cnn":
                self.encoder = nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling = config["pooling_style"]
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)[0]  # (batch_size, seq_len, hidden_size)
        else:
            x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
            x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        if self.pooling == "max":
            x = torch.max(x, dim=1)[0]
        else:
            x = torch.mean(x, dim=1)

        predict = self.classify(x)

        if target is not None:
            target = target.squeeze()
            return self.loss(predict, target)
        else:
            return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    lr = config["learning_rate"]

    if optimizer == "adam":
        return Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type")
