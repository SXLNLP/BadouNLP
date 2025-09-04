# -*- coding: utf-8 -*-
# 主要修改了model的部分

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizerFast
from peft import LoraConfig, get_peft_model

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()

        bilstm_hidden_size = config["bilstm_hidden_size"]
        dropout = config["bilstm_dropout"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        # self.bert = BertModel.from_pretrained(config["bert_model"])
        self.bert = BertModel.from_pretrained(config["bert_path"])

        # 加载Lora配置
        lora_config = LoraConfig(
            r=8,  # LoRA的秩
            lora_alpha=16,  # 缩放因子
            target_modules=["query", "key", "value"],  # 对BERT的query,key,value矩阵应用LoRA
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classify"]  # 除了LoRA外，正常训练分类层
        )

        # bert转lora
        self.bert = get_peft_model(self.bert, lora_config)
        self.bert.print_trainable_parameters()  # 打印可训练参数数量

        hidden_size = config["hidden_size"]
        class_num = config["class_num"]

        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, target=None):  # 添加attention_mask参数
        # x = self.embedding(x)  # input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      # input shape:(batch_size, sen_len, input_dim)
        output = self.bert(x, attention_mask=attention_mask)  # 传入attention_mask
        x = output[0]
        predict = self.classify(x)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                if attention_mask is None:
                    mask = attention_mask.bool()
                else:
                    mask = torch.ones(x.size()[:2], dtype=torch.bool, device=x.device)
                # return self.crf_layer._viterbi_decode(predict)
                return self.crf_layer.decode(predict, mask=mask)
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
