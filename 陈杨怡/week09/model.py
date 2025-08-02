# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
"""
建立网络模型结构
"""

from transformers import BertForTokenClassification

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(config["bert_path"], num_labels=config["class_num"])

    def forward(self, input_ids, attention_mask, target=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=target)
        return outputs.loss if target is not None else outputs.logits


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
