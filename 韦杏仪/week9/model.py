# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
# from torchcrf import CRF
from TorchCRF import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]

        self.bert = BertModel.from_pretrained(config["bert_path"])  # 加载Bert模型
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        # self.classify = nn.Linear(hidden_size * 2, class_num)
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        # self.crf_layer = CRF(class_num, batch_first=True)
        self.crf_layer = CRF(class_num)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        # predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
        attention_mask = (x != 0).float()
        outputs = self.bert(x, attention_mask=attention_mask)  # 使用Bert模型
        predict = self.classify(outputs[0])

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                loss = - self.crf_layer(predict, target, mask)
                # 确保 loss 是标量
                if loss.numel() > 1:
                    loss = loss.mean()
                return loss
            else:
                # (number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                # return self.crf_layer.decode(predict)
                return self.crf_layer(predict)  # 尝试用forward替代decode
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