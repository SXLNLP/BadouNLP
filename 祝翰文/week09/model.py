# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel, BertConfig
from TorchCRF import CRF

"""
建立基于BERT的网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(config["bert_path"])
        bert_config = BertConfig.from_pretrained(config["bert_path"])
        hidden_size = bert_config.hidden_size

        # 分类层
        self.classify = nn.Linear(hidden_size, config["class_num"])

        # CRF层（如果使用）
        self.use_crf = config["use_crf"]
        if self.use_crf:
            self.crf_layer = CRF(config["class_num"], batch_first=True)

        # 损失函数
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        # 通过BERT获取输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 获取最后一层的隐藏状态 [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state

        # 分类层
        logits = self.classify(sequence_output)  # [batch_size, seq_len, class_num]

        if labels is not None:
            if self.use_crf:
                # 创建mask（忽略padding位置）
                mask = attention_mask.bool()
                # 计算CRF损失
                loss = -self.crf_layer(logits, labels, mask=mask, reduction="mean")
                return loss
            else:
                # 调整维度计算交叉熵损失
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.size(-1))[active_loss]
                active_labels = labels.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
        else:
            if self.use_crf:
                mask = attention_mask.bool()
                return self.crf_layer.decode(logits, mask=mask)
            else:
                # 直接返回预测结果
                return torch.argmax(logits, dim=-1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    # 为BERT层和分类层设置不同的学习率
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.bert.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': learning_rate
        },
        {
            'params': [p for n, p in model.bert.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': learning_rate
        },
        {
            'params': model.classify.parameters(),
            'lr': learning_rate * 5  # 分类层使用更高的学习率
        }
    ]

    if config["use_crf"]:
        optimizer_grouped_parameters.append({
            'params': model.crf_layer.parameters(),
            'lr': learning_rate * 5
        })

    if optimizer == "adam":
        return Adam(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(optimizer_grouped_parameters, lr=learning_rate, momentum=0.9)


if __name__ == "__main__":
    from config import Config

    # 在配置中添加bert_path
    Config["bert_path"] = "bert-base-chinese"
    model = TorchModel(Config)
