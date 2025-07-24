# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from torch.optim import Adam, SGD


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
        # BERT 输出 shape: (batch_size, seq_len, hidden_size)
        # outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        # sequence_output = outputs.last_hidden_state
        sequence_output = self.bert(input_ids=x, attention_mask=attention_mask)[0]
        predict = self.classify(sequence_output)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    if config["optimizer"] == "adam":
        return Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        return SGD(model.parameters(), lr=config["learning_rate"])
