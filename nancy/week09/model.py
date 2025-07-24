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
        self.bert_like = BertModel.from_pretrained(config["bert_path"])
        self.dropout = nn.Dropout(self.bert_like.config.hidden_dropout_prob)
        class_num = config["class_num"]
        self.num_labels = config["class_num"]
        if config["recurrent"] == "lstm":
            self.recurrent_layer = nn.LSTM(self.bert_like.config.hidden_size,
                                      self.bert_like.config.hidden_size // 2,
                                      batch_first=True,
                                      bidirectional=True,
                                      num_layers=1
                                      )
        elif config["recurrent"] == "gru":
            self.recurrent_layer = nn.GRU(self.bert_like.config.hidden_size,
                                      self.bert_like.config.hidden_size // 2,
                                      batch_first=True,
                                      bidirectional=True,
                                      num_layers=1
                                      )
        else:
            assert False

        self.classifier = nn.Linear(self.bert_like.config.hidden_size, config["class_num"])
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        # self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids=None,
                      attention_mask=None,
                      labels=None):
        #(num_sentence, sentence_length)
        if input_ids is None or input_ids.shape[1] == 0:
            raise ValueError("input_ids shape error: {}".format(input_ids.shape))
        outputs = self.bert_like(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        recurrent_output, _ = self.recurrent_layer(sequence_output)
        output = self.classifier(recurrent_output)

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1) 
                return - self.crf_layer(output, labels, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(output.view(-1, output.shape[-1]), labels.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(output)
            else:
                return output


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
