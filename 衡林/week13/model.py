# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig
from peft import LoraConfig, get_peft_model
from typing import Optional


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]
        self.freeze_bert = config.get("freeze_bert", False)

        # Initialize BERT with custom configuration
        bert_config = BertConfig.from_pretrained(
            config.get("bert_path", "bert-base-chinese"),
            output_hidden_states=True,
            return_dict=True
        )

        # Customize BERT architecture if specified
        if "bert_num_layers" in config:
            bert_config.num_hidden_layers = config["bert_num_layers"]
        if "bert_hidden_size" in config:
            bert_config.hidden_size = config["bert_hidden_size"]

        # Load BERT model
        self.bert = BertModel.from_pretrained(
            config.get("bert_path", "bert-base-chinese"),
            config=bert_config
        )

        # Freeze BERT parameters if needed
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Add LoRA if specified
        if config.get("use_lora", False):
            lora_config = LoraConfig(
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                target_modules=["query", "key", "value"],
                lora_dropout=config.get("lora_dropout", 0.1),
                bias="none",
                modules_to_save=["classify"]  # Ensure classification layer is trained
            )
            self.bert = get_peft_model(self.bert, lora_config)
            self.bert.print_trainable_parameters()

        # Classification layer
        bert_hidden_size = self.bert.config.hidden_size
        self.classify = nn.Linear(bert_hidden_size, class_num)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(bert_hidden_size)

        # Dropout
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.1))

        # CRF layer
        if self.use_crf:
            self.crf_layer = CRF(class_num, batch_first=True)
        else:
            # Loss function (with class weights if specified)
            weight = config.get("class_weights", None)
            if weight is not None:
                weight = torch.tensor(weight, dtype=torch.float)
            self.loss = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=-1,
                label_smoothing=config.get("label_smoothing", 0.0)
            )

    def forward(self,
                x: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (x != 0).long()

        # BERT forward pass
        bert_output = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use last hidden states or concatenate last few layers
        if self.config.get("use_all_layers", False):
            # Concatenate last 4 layers
            hidden_states = bert_output.hidden_states
            sequence_output = torch.cat(hidden_states[-4:], dim=-1)
            sequence_output = self.layer_norm(sequence_output)
        else:
            sequence_output = bert_output.last_hidden_state

        sequence_output = self.dropout(sequence_output)

        # Classification
        logits = self.classify(sequence_output)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return -self.crf_layer(logits, target, mask, reduction="mean")
            else:
                # Flatten the tensors for cross entropy loss
                return self.loss(
                    logits.view(-1, logits.shape[-1]),
                    target.view(-1)
                )
        else:
            if self.use_crf:
                # CRF decode returns list of lists
                paths = self.crf_layer.decode(logits)
                # Convert to tensor format expected by evaluation
                batch_size, seq_len = logits.shape[:2]
                result = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
                for i, path in enumerate(paths):
                    path_len = min(len(path), seq_len)
                    result[i, :path_len] = torch.tensor(path[:path_len], device=logits.device)
                return result
            else:
                return torch.argmax(logits, dim=-1)


def choose_optimizer(config, model):
    optimizer_config = config["optimizer"]
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.01)

    # Get parameters to optimize (filter out frozen ones)
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_config == "adam":
        return Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_config == "adamw":
        return AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_config == "sgd":
        return SGD(
            params,
            lr=learning_rate,
            momentum=config.get("momentum", 0.9),
            nesterov=config.get("nesterov", True)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config}")


if __name__ == "__main__":
    from config import Config

    config = Config
    config.update({
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "freeze_bert": True,
        "dropout_rate": 0.2,
        "use_all_layers": False,
        "label_smoothing": 0.1,
        "weight_decay": 0.01
    })
    model = TorchModel(config)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")