# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.use_bert = config["use_bert"]
        if self.use_bert:
            # 加载BERT模型
            self.bert = BertModel.from_pretrained(config["bert_path"])
            hidden_size = config["bert_hidden_size"]

            # 冻结BERT参数(如果不微调)
            if not config["fine_tune_bert"]:
                for param in self.bert.parameters():
                    param.requires_grad = False
        else:
            # 保留原有的LSTM实现
            self.embedding = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=0)
            self.layer = nn.LSTM(config["hidden_size"], config["hidden_size"],
                                 batch_first=True, bidirectional=True,
                                 num_layers=config["num_layers"])
            hidden_size = config["hidden_size"] * 2

        # 分类层
        self.classify = nn.Linear(hidden_size, config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):

        x, _ = self.bert(x)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict
        # if self.use_bert:
        #     # BERT前向传播
        #     outputs = self.bert(x)
        #     x = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        # else:
        #     # 原有的LSTM前向传播
        #     x = self.embedding(x)
        #     x, _ = self.layer(x)
        #
        # predict = self.classify(x)
        #
        # # 保持原有的CRF/CrossEntropy部分不变
        # if target is not None:
        #     if self.use_crf:
        #         mask = target.gt(-1)
        #         return -self.crf_layer(predict, target, mask, reduction="mean")
        #     else:
        #         return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        # else:
        #     if self.use_crf:
        #         return self.crf_layer.decode(predict)
        #     else:
        #         return predict


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