# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # max_length = config["max_length"]
        # class_num = config["class_num"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])  # 新增tokenizer
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        # self.crf_layer = CRF(class_num, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        # 计算loss时忽略padding部分
        self.loss = nn.CrossEntropyLoss(ignore_index=-100) #loss采用交叉熵损失 假设pad_token_id=0
        self.to(self.device)  # 确保所有参数转移到正确设备

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 设备安全处理
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classify(outputs[0])

        if labels is not None:
            labels = labels.to(self.device)
            loss_mask = (labels != -100)
            return self.loss(logits[loss_mask], labels[loss_mask])
        return logits


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
    config = {
        "bert_path": "bert-base-chinese",
        "optimizer": "adam",
        "learning_rate": 5e-5
    }
    device = next(model.parameters()).device
    # 示例输入
    input_ids = torch.LongTensor([[101, 234, 345, 456, 0]]).to(device)  # 假设0是padding
    attention_mask = torch.tril(torch.ones((1, 5, 5))).to(device)  # 下三角自回归mask
    labels = torch.LongTensor([[-100, 345, 456, 567, 0]]).to(device)  # 下一个token的标签

    loss = model(input_ids, attention_mask, labels)
    print(loss)