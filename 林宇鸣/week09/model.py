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
        self.config = config
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # # 这里用的是一个双向LSTM
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict = False)  # Load BERT model
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.classify2 = nn.Linear(self.bert.config.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Loss using cross-entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
       if(self.config["model_type"] == "bert"):
           x = self.bert(x)[0] #input shape:(batch_size, sen_len, input_dim)
           predict = self.classify2(x)
       else:
           x = self.embedding(x)
           x, _ = self.layer(x) #input shape:(batch_size, sen_len, input_dim)
           predict = self.classify(x) #output_shape:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

       if target is not None:
           if self.use_crf:
               # 用crf的话计算的是crf loss
               mask = target.gt(-1)
               # 取相反数才能作为loss
               # 不同的库实现方式不同
               return - self.crf_layer(predict, target, mask, reduction="mean")
           else:
               # 否则的话就用交叉熵来计算loss
               # (number, class_num), (number)
               return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
       else:
           if self.use_crf:
               return self.crf_layer.decode(predict)
           else:
               # 输出发射矩阵
               return predict

        # if target is not None:



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