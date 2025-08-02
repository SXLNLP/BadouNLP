# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from crf import CRF
from transformers import BertModel

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classify = nn.Linear(768, config["class_num"])
        self.use_crf = config["use_crf"]
        if self.use_crf:
            self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_ids, attention_mask, labels=None):
        # 添加 return_dict=True 确保返回字典格式的结果
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 正确获取 last_hidden_state
        sequence_output = outputs.last_hidden_state

        logits = self.classify(sequence_output)

        if labels is not None:
            if self.use_crf:
                mask = (labels != -100)  # 注意标签中-100表示忽略
                mask = mask.bool()  # 使用 bool 类型而不是 byte
                loss = -self.crf_layer(logits, labels, mask=mask, reduction="mean")
                return loss
            else:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])[active_loss]
                active_labels = labels.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
        else:
            if self.use_crf:
                mask = attention_mask.bool()
                return self.crf_layer.decode(logits, mask=mask)
            else:
                return logits


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
