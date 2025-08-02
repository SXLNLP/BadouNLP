# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer, AutoConfig

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])  # 加载预训练BERT模型
        self.hidden_size = self.bert.config.hidden_size
        self.num_labels = config["class_num"]
        self.use_crf = config["use_crf"]

        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.crf_layer = CRF(self.num_labels, batch_first=True)

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1)
                loss = -self.crf_layer(emissions, labels, mask=mask, reduction='mean')
            else:
                loss = self.loss(emissions.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            if self.use_crf:
                return self.crf_layer.decode(emissions, mask=attention_mask.bool())
            else:
                return emissions

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
