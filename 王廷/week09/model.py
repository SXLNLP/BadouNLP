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
        self.bert = BertModel.from_pretrained(config["bert_path"])
        hidden_size = self.bert.config.hidden_size
        class_num = config["class_num"]
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None, attention_mask=None):
        # 修改这里：直接获取第一个元素作为序列输出
        sequence_output = self.bert(x, attention_mask=attention_mask)[0]
        predict = self.classify(sequence_output)

        if target is not None:
            if self.use_crf:
                mask = attention_mask.bool()
                return -self.crf_layer(predict, target, mask=mask, reduction="mean")
            else:
                active_loss = attention_mask.view(-1) == 1
                active_logits = predict.view(-1, predict.shape[-1])[active_loss]
                active_labels = target.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
        else:
            if self.use_crf:
                mask = attention_mask.bool()
                return self.crf_layer.decode(predict, mask=mask)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)