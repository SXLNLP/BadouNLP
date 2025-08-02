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
        # todo
        #  词嵌入层和layer直接换成Bert模型，再送入分类层映射到类别维度
        #  无需使用crf了(config里设置就好)
        if config["model_type"] == "bert":
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            self.classify = nn.Linear(hidden_size, class_num)
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True,
                                   num_layers=num_layers)
            # bidirectional=True===>双向LSTM，正向输入和逆向输入，然后将两个hidden拼接，输出维度为hidden_size * 2
            self.classify = nn.Linear(hidden_size * 2, class_num)  # 所以输入纬度是hidden_size * 2，然后再映射到类别维数class_num

        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.config["model_type"] == "bert":
            x, _ = self.encoder(x)
        else:
            x = self.embedding(x)  #input shape:(batch_size, sen_len)
            x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)

        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:  # 训练阶段，可以出表总结‼️
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)  # 维特比解码
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