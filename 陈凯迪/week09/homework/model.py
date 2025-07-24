# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel


class TorchModel(BertPreTrainedModel):
    def __init__(self, config, model_config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, model_config["class_num"])
        self.crf_layer = CRF(model_config["class_num"], batch_first=True)
        self.use_crf = model_config["use_crf"]
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            if self.use_crf:
                mask = (input_ids != 0).byte()  # 创建mask，忽略padding
                loss = -self.crf_layer(logits, labels, mask=mask, reduction="mean")
                return loss
            else:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fn(active_logits, active_labels)
                return loss
        else:
            if self.use_crf:
                mask = (input_ids != 0).byte()
                return self.crf_layer.decode(logits, mask=mask)
            else:
                return torch.argmax(logits, dim=-1)