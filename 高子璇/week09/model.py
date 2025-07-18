# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from TorchCRF import CRF

class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_crf = config["use_crf"]
        self.bert = BertModel.from_pretrained(config["bert_path"])
        hidden_size = config["hidden_size"]
        self.classifier = nn.Linear(hidden_size, config["class_num"])
        self.crf = CRF(config["class_num"], batch_first=True)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids      : [batch, seq_len]
        attention_mask : [batch, seq_len]
        labels         : [batch, seq_len]  或 None
        """
        # bert_out = self.bert(input_ids=input_ids,
        #                      attention_mask=attention_mask)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_out = bert_out.last_hidden_state   # [batch, seq_len, 768]
        logits = self.classifier(sequence_out)        # [batch, seq_len, 9]

        # if labels is not None:
        #     if self.use_crf:
        #         mask = labels.gt(-1)
        #         return -self.crf(logits, labels, mask=mask, reduction='mean')
        #     else:
        #         return self.loss_fn(logits.view(-1, logits.size(-1)),
        #                             labels.view(-1))
        if labels is not None:
            if self.use_crf:
                # 使用 attention_mask 作为 mask（更可靠）
                mask = attention_mask.bool()

                return -self.crf(logits, labels, mask=mask, reduction='mean')
            else:
                return self.loss_fn(logits.view(-1, logits.size(-1)),
                                    labels.view(-1))
        else:
            if self.use_crf:
                return self.crf.decode(logits, mask=attention_mask)
            else:
                return logits


def choose_optimizer(config, model):
    from torch.optim import Adam
    return Adam(model.parameters(), lr=config["learning_rate"])
