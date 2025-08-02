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
        self.loss = nn.CrossEntropyLoss(ignore_index=0) #loss采用交叉熵损失 假设pad_token_id=0
        self.to(self.device)  # 确保所有参数转移到正确设备

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,  # 使用自定义的自回归mask
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )  # output shape:(batch_size, sen_len, input_dim)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        logits = self.classify(sequence_output)  # output shape:(batch_size, sen_len, vocab_size)
        if labels is not None:
            # 创建padding mask（假设padding_id=0）
            padding_mask = (input_ids != 0)  # [batch_size, seq_len]
            # 只计算非padding位置的loss
            active_loss = padding_mask.view(-1)   # [batch_size, seq_len] -> 需要调整
            active_logits = logits.view(-1, self.bert.config.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss(active_logits, active_labels)
            return loss
        else:
            return torch.softmax(logits, dim=-1)



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

    # 示例输入
    input_ids = torch.LongTensor([[101, 234, 345, 456, 0]])  # 假设0是padding
    attention_mask = torch.tril(torch.ones((1, 5, 5)))  # 下三角自回归mask
    labels = torch.LongTensor([[234, 345, 456, 567, 0]])  # 下一个token的标签

    loss = model(input_ids, attention_mask, labels)
    print(loss)