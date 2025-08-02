# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
from transformers import BertTokenizerFast

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
        self.pooling_style = config["pooling_style"]
        self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False,
                                           num_hidden_layers=num_layers)

        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        # 增加Dropout层防止过拟合
        self.dropout = nn.Dropout(0.1)
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=-1)  # loss采用交叉熵损失 ignore_index=-1 是一种常见的训练技巧，尤其在序列任务中，通过忽略填充标签来优化模型学习过程

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, target=None):
        # sequence_output:batch_size, max_len, hidden_size
        # pooler_output:batch_size, hidden_size

        x = self.encoder(x, attention_mask=attention_mask)[0]  # 直接取序列输出
        x = self.dropout(x)  # 增加dropout
        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]

        predict = self.classify(x)  # ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                # 生成布尔掩码：标记所有大于-1的元素为True，其余为False
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 1.predict.view(-1, predict.shape[-1]) -1 batch_size * sen_len
                # 2.target.view(-1) - 将目标张量展平为一维
                # 3.self.loss() - 调用损失函数（如交叉熵损失）计算两个张量间的误差
                # 主要完成模型输出与标签的维度对齐和损失计算。
                # (number, class_num), (number)
                # 检查模型forward返回值与标签形状：
                # 假设标签原始形状为(batch_size, seq_len)，需确保predict形状为(batch_size, seq_len, num_classes)
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
