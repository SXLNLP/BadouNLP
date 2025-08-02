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
        self.use_crf = config["use_crf"]
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=True)
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # (batch, seq_len, hidden)
        logits = self.classify(sequence_output)

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1)
                return -self.crf_layer(logits, labels, mask, reduction="mean")
            else:
                return self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(logits)
            else:
                return torch.argmax(logits, dim=-1)

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