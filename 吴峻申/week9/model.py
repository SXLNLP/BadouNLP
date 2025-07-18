import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF


class TorchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.class_num)
        self.crf = CRF(config.class_num, batch_first=True)
        self.use_crf = config.use_crf
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # 使用-100忽略填充标签

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if labels is not None:
            if self.use_crf:
                # 创建用于CRF的掩码
                mask = attention_mask.bool()
                # 过滤掉填充标签
                active_labels = labels.clone()
                active_labels[labels == -100] = 0  # 将-100替换为0
                loss = -self.crf(logits, active_labels, mask=mask, reduction='mean')
                return loss
            else:
                # 处理交叉熵损失
                loss_mask = (labels != -100)  # 忽略填充标签
                active_logits = logits[loss_mask]
                active_labels = labels[loss_mask]
                loss = self.loss(active_logits, active_labels)
                return loss
        else:
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
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
