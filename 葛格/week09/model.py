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
        self.use_bert = config.get("use_bert", False)
        self.class_num = config["class_num"]

        if self.use_bert:
            self.bert = BertModel.from_pretrained(config["bert_path"])
            hidden_size = self.bert.config.hidden_size
        else:
            vocab_size = config["vocab_size"] + 1
            hidden_size = config["hidden_size"]
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=config["num_layers"])
            hidden_size *= 2

        self.classify = nn.Linear(hidden_size, self.class_num)
        self.crf_layer = CRF(self.class_num, batch_first=True)
        # 定义交叉熵损失，忽略-100标签
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # 定义pad_label_id：替换 -100 用的，必须是合法标签id，比如你的O标签8
        self.pad_label_id = config.get("pad_label_id", 8)  # 默认8，可根据实际调整

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None, attention_mask=None):
        if self.use_bert:
            x = self.bert(input_ids=x, attention_mask=attention_mask)[0] # [batch, seq_len, hidden]
        else:
            x = self.embedding(x)
            x, _ = self.layer(x)

        predict = self.classify(x)  # [batch, seq_len, class_num]

        if target is not None:
            if self.use_crf:
                # CRF 的 mask：为 True 表示有效 token。忽略 pad（不包括 label 中的 -100）
                mask = target.gt(-1)
                mask[:, 0] = True
                # 替换 -100 为 pad_label_id
                target = torch.where(target == -100, torch.tensor(self.pad_label_id, device=target.device), target)
                return -self.crf_layer(predict, target, mask=mask, reduction="mean")

            else:
                # 交叉熵要展平，忽略 -100 标签
                return self.loss_fn(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                # 解码时传入 attention_mask（为 True 的部分才处理）
                return self.crf_layer.decode(predict, mask=attention_mask.bool())
            else:
                return predict  # [batch, seq_len, class_num]


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
    print(type(model))