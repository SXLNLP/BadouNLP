# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel, BertConfig


class BertNER(BertPreTrainedModel):
    def __init__(self, config, use_crf=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_crf = use_crf

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if self.use_crf:
            self.crf = CRF(config.num_labels, batch_first=True)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            if self.use_crf:
                mask = labels.ne(-1)  # -1是padding的标签
                mask[:, 0] = 1  # 确保第一个token有效
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            else:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            return loss
        else:
            if self.use_crf:
                return self.crf.decode(logits, mask=attention_mask.ne(0))
            else:
                return torch.argmax(logits, dim=-1)


class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_config = BertConfig.from_pretrained(
            config["bert_path"],
            num_labels=config["class_num"]
        )
        self.bert_ner = BertNER.from_pretrained(
            config["bert_path"],
            config=bert_config,
            use_crf=config["use_crf"]
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.bert_ner(input_ids, attention_mask, labels)


def choose_optimizer(config, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': config["weight_decay"],
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]

    if config["optimizer"] == "adamw":
        return AdamW(optimizer_grouped_parameters,
                     lr=config["learning_rate"],
                     eps=1e-8)
    elif config["optimizer"] == "adam":
        return Adam(optimizer_grouped_parameters,
                    lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        return SGD(optimizer_grouped_parameters,
                   lr=config["learning_rate"],
                   momentum=0.9)
