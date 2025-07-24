# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]
        
        # Load BERT model with custom configuration for small dataset
        bert_config = BertConfig.from_pretrained(config.get("bert_path", "bert-base-chinese"))
        if "bert_num_layers" in config:
            bert_config.num_hidden_layers = config["bert_num_layers"]
        
        self.bert = BertModel.from_pretrained(
            config.get("bert_path", "bert-base-chinese"), 
            config=bert_config
        )
        
        # Get BERT hidden size
        bert_hidden_size = self.bert.config.hidden_size
        
        # Classification layer
        self.classify = nn.Linear(bert_hidden_size, class_num)
        
        # CRF layer
        self.crf_layer = CRF(class_num, batch_first=True)
        
        # Loss function
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (x != 0).long()
        
        # BERT forward pass
        bert_output = self.bert(input_ids=x, attention_mask=attention_mask)
        sequence_output = bert_output[0]  # (batch_size, seq_len, hidden_size)
        
        # Classification
        predict = self.classify(sequence_output)  # (batch_size, seq_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                # CRF decode returns list of lists, convert to tensor for evaluation
                paths = self.crf_layer.decode(predict)
                # Convert to tensor format expected by evaluation
                batch_size, seq_len = predict.shape[:2]
                result = torch.zeros(batch_size, seq_len, dtype=torch.long, device=predict.device)
                for i, path in enumerate(paths):
                    path_len = min(len(path), seq_len)
                    result[i, :path_len] = torch.tensor(path[:path_len], device=predict.device)
                return result
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