# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel

"""
使用BERT建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask, target=None):
        # BERT编码 - 明确设置return_dict=True来确保返回格式
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # 获取序列输出
        sequence_output = bert_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        # 分类层
        predict = self.classify(sequence_output)  # (batch_size, seq_len, class_num)

        if target is not None:
            if self.use_crf:
                # 使用attention_mask作为CRF的mask
                mask = attention_mask.bool()
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 只计算非padding位置的loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = predict.view(-1, predict.shape[-1])[active_loss]
                active_labels = target.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
        else:
            if self.use_crf:
                mask = attention_mask.bool()
                return self.crf_layer.decode(predict, mask)
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