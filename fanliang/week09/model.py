# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
import os

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # 创建一个双向LSTM层
        # input_size: LSTM的输入维度
        # hidden_size: LSTM的隐藏维度
        # batch_first: 如果为True，则输入和输出的数据格式为（batch_size, seq_len, feature）
        # bidirectional: 如果为True，则LSTM为双向
        # num_layers: LSTM的层数
        self.layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert = BertModel.from_pretrained(os.path.dirname(os.path.abspath(__file__))+"/../../models/bert-base-chinese", return_dict=False)
        #注意上面LSTM使用的bidirectional: 如果为True，则LSTM为双向，则生成的张量维度是单项的2倍，所以要与bert模型做区分
        if(config["model"] == "LSTM"): 
            self.classify = nn.Linear(hidden_size * 2, class_num)
        else:
            self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if(self.config["model"] != "bert"):
            x = self.embedding(x)  #input shape:(batch_size, sen_len)
            x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        else:
            x, y = self.bert(x)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
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