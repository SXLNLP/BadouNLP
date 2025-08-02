# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertForTokenClassification, AutoConfig

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        model_config = AutoConfig.from_pretrained(config["pretrain_model_path"], num_labels=config["num_labels"])
        self.bert = BertForTokenClassification.from_pretrained(
            config["pretrain_model_path"],
            config=model_config
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["optimizer"] = "adamw"
    model = TorchModel(Config)
    input_ids = torch.LongTensor([[0,1,2,3,4,100,6,7,8], [0,4,3,2,1,100,8,7,6]])
    labels = torch.LongTensor([[1], [0]])
    print(model(input_ids, labels=labels))
