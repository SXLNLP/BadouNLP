# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]  # 指定使用的层数

        # 加载预训练的BERT配置
        bert_config = BertConfig.from_pretrained(config["bert_path"])

        # 创建一个新的BERT模型实例，并只加载前3层的编码器
        self.bert = BertModel(bert_config)
        self.bert.load_state_dict(torch.load(f"{config['bert_path']}/pytorch_model.bin"), strict=False)

        # 只保留前3层的编码器
        self.bert.encoder.layer = self.bert.encoder.layer[:num_layers]

        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, labels=None):
        outputs = self.bert(x, attention_mask=attention_mask)
        sequence_output = outputs[0]  # 提取第一个元素作为sequence_output
        predict = self.classify(sequence_output)  # output: (batch_size, seq_len, num_tags)

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1).bool()  # 将mask转换为布尔类型
                return - self.crf_layer(predict, labels, mask, reduction="mean")
            else:
                return self.loss_fn(predict.view(-1, predict.shape[-1]), labels.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict



