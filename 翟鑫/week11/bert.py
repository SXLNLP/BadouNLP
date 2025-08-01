# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络模型结构
"""


class BertGeneratorModel(nn.Module):
    def __init__(self, config):
        super(BertGeneratorModel, self).__init__()

        self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        self.vocab_size = config["vocab_size"]

        self.classify = nn.Linear(hidden_size, self.vocab_size)
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失
        self.pooling = nn.AvgPool1d(self.vocab_size)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x,target=None):  # x 输入时加入 问题长度 [B,T,E] B个句子为一个batch, 一个句子T个词，每个词E个维度. target-> [B, [x,T]] x个词作为问题，T为target
        # 左边mask，右边是e
        attention_mask = make_qa_causal_mask(x)
        x = self.encoder(x, attention_mask=attention_mask)
        predict = self.classify(x[0])  # output shape: probability(batch_size, vocab_size)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)



def make_qa_causal_mask(input_ids,
                        pad_token_id=0,
                        sep_token_id=102,   # BERT 的 [SEP]
                        question_len=None):
    """
    input_ids:  (batch, seq_len)  已经 tokenize + pad
    question_len: 如果已知，可直接传进来；否则用第一个 [SEP] 位置自动推断
    return: attention_mask (batch, seq_len, seq_len)
    """
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    # 1. 初始化全 1，再按规则改
    mask = torch.ones(bsz, seq_len, seq_len, dtype=torch.long, device=device)

    # 2. 处理 padding
    pad_mask = (input_ids != pad_token_id).long()  # (bsz, seq_len)
    mask = mask * pad_mask.unsqueeze(-1) * pad_mask.unsqueeze(1)

    # 3. 找到每个样本的问题长度（不含回答）
    if question_len is None:
        # 用第一个 [SEP] 作为分界
        sep_pos = (input_ids == sep_token_id).long().argmax(dim=1)  # (bsz,)
        question_len = sep_pos + 1  # 把 [SEP] 也含进去

    # 4. 回答区段：下三角
    for b in range(bsz):
        start = question_len[b]     # 回答开始位置
        causal = torch.tril(torch.ones(seq_len - start, seq_len - start,
                                       dtype=torch.long, device=device))
        mask[b, start:, start:] = causal

    return mask


if __name__ == "__main__":
    from config import Config

    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))
