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
        super().__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])

        # 冻结BERT参数（可选）
        if config.get("freeze_bert", False):
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classify = nn.Linear(config["bert_hidden_size"], config["class_num"])
        self.dropout = nn.Dropout(0.3)  # 添加Dropout防止过拟合

        # 延迟加载CRF
        self.use_crf = config["use_crf"]
        if self.use_crf:
            self.crf_layer = CRF(config["class_num"], batch_first=True)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=8)  # 忽略'O'标签

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classify(sequence_output)

        if labels is not None:
            # 检查标签值是否在合法范围内
            invalid_labels = (labels < 0) | (labels > 8)
            if invalid_labels.any():
                print(f"发现非法标签值: {labels[invalid_labels].unique().tolist()}")
                # 将非法标签设为8（O标签）
                labels = torch.where(invalid_labels, torch.full_like(labels, 8), labels)

            if self.use_crf:
                mask = attention_mask.bool()
                return -self.crf_layer(logits, labels, mask, reduction="mean")
            else:
                # 计算有效位置损失
                active_loss = (labels != 8) & (attention_mask == 1)
                active_logits = logits[active_loss]
                active_labels = labels[active_loss]

                if active_labels.numel() > 0:
                    return self.loss(active_logits, active_labels)
                return torch.tensor(0.0, device=logits.device)
        else:
            if self.use_crf:
                return self.crf_layer.decode(logits, mask=attention_mask.bool())
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
