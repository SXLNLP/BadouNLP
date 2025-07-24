# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig, AutoTokenizer
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert = BertModel.from_pretrained(config["bert_path"])
        bert_config = self.bert.config
        self.classify = nn.Linear(bert_config.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids,attention_mask, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        outputs = self.bert(input_ids,attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        predict = self.classify(sequence_output) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = attention_mask.bool()
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                active_loss = attention_mask.view(-1) == 1
                active_logits = predict.view(-1, predict.shape[-1])
                active_labels = target.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # 分词处理（启用单词级分割）
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True  # 关键参数：保持单词级输入
    )

    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        # 获取token对应的原始单词索引
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        adjusted_labels = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:  # 特殊token：[CLS], [SEP], [PAD]
                adjusted_labels.append(-100)  # 忽略标签
            elif word_idx != prev_word_idx:  # 新单词的第一个token
                adjusted_labels.append(label_seq[word_idx])  # 原始标签
            else:  # 同一单词的后续token
                prev_label = label_seq[word_idx]
                # 转换B-标签为I-标签
                if prev_label.startswith("B-"):
                    adjusted_labels.append(prev_label.replace("B-", "I-"))
                else:
                    adjusted_labels.append(prev_label)
            prev_word_idx = word_idx

        labels.append(adjusted_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



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
