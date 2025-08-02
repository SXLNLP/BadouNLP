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
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        if config["use_bert"]:
            self.bert = BertModel.from_pretrained(config["bert_path"])
            self.classify = nn.Linear(768, class_num)
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
            self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.use_bert = config["use_bert"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, target=None):
        if self.use_bert:
            x = self.bert(x, attention_mask=attention_mask, return_dict=True)
            x = x.last_hidden_state
        else:
            x = self.embedding(x)
            x, _ = self.layer(x)
        predict = self.classify(x)

        if self.use_bert:
            if target is not None:
                if self.use_crf:
                    mask = (target != -100).bool()
                    mask = mask & (attention_mask.bool())  # 结合attention_mask
                    mask[:, 0] = True  # 确保第一个token有效
                    print(x, target)
                    return - self.crf_layer(predict, target, mask, reduction="mean")
                else:
                    flag = attention_mask.view(-1) == 1
                    return self.loss(predict.view(-1, predict.shape[-1])[flag], target.view(-1)[flag])
            else:
                if self.use_crf:
                    return self.crf_layer.decode(predict, attention_mask)
                else:
                    return predict
        elif target is not None:
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
