# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.class_num = config["class_num"]
        # 分类层，将BERT输出映射到标签空间
        self.classify = nn.Linear(self.bert.config.hidden_size, self.class_num)
        self.use_crf = config["use_crf"]
        
        # 如果使用CRF，初始化CRF层
        if self.use_crf:
            self.crf = CRF(self.class_num, batch_first=True)
        
        # 交叉熵损失函数，忽略-100标签
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 获取BERT最后一层的输出
        sequence_output = outputs.last_hidden_state
        
        # 通过分类层得到预测标签
        logits = self.classify(sequence_output)
        
        if labels is not None:
            if self.use_crf:
                # 创建 mask（True表示有效位置）
                mask = attention_mask.bool()
                
                # 复制 labels，避免修改原始数据
                labels_for_crf = labels.clone()
                labels_for_crf[labels_for_crf == -100] = 0  # 替换为合法标签索引，如0

                # CRF 前向计算（最大似然）
                loss = -self.crf(logits, labels_for_crf, mask=mask, reduction='mean')
            else:
                # 使用交叉熵损失
                loss = self.loss(logits.view(-1, self.class_num), labels.view(-1))
            return loss
        else:
            # 推理模式（预测标签）
            if self.use_crf:
                mask = attention_mask.bool()
                return self.crf.decode(logits, mask=mask)
            else:
                return torch.argmax(logits, dim=-1)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
