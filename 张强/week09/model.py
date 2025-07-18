# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立基于 BERT 的 NER 网络模型结构
"""

class BertCRFModel(nn.Module):
    def __init__(self, config):
        super(BertCRFModel, self).__init__()
        # hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        # max_length = config["max_length"]
        # class_num = config["class_num"]
        # num_layers = config["num_layers"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        # self.classify = nn.Linear(hidden_size * 2, class_num)
        self.bert = BertModel.from_pretrained(
            config["bert_path"],return_dict=True,
            output_hidden_states=True
        )

        # 获取 BERT 配置
        self.hidden_size = self.bert.config.hidden_size
        self.class_num = config["class_num"]
        self.use_crf = config["use_crf"]

        # 分类层
        self.classify = nn.Linear(self.hidden_size, self.class_num)

        # Dropout 层
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.3))

        # CRF 层（如果使用）
        if self.use_crf:
            self.crf_layer = CRF(self.class_num, batch_first=True)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, target=None):
    #     x = self.embedding(x)  #input shape:(batch_size, sen_len)
    #     x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
    #     predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
    #
    #     if target is not None:
    #         if self.use_crf:
    #             mask = target.gt(-1)
    #             return - self.crf_layer(predict, target, mask, reduction="mean")
    #         else:
    #             #(number, class_num), (number)
    #             return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
    #     else:
    #         if self.use_crf:
    #             return self.crf_layer.decode(predict)
    #         else:
    #             return predict
    def forward(self, input_ids, attention_mask, target=None):
        # BERT 前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 获取最后一层隐藏状态
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # 应用 dropout
        sequence_output = self.dropout(sequence_output)

        # 分类层
        logits = self.classify(sequence_output)  # 形状: (batch_size, seq_len, class_num)

        if target is not None:
            if self.use_crf:
                # 确保标签在有效范围内 (0 到 class_num-1)
                valid_mask = (target >= 0) & (target < self.class_num)

                # 创建CRF掩码（排除填充位置）
                crf_mask = attention_mask.bool() & valid_mask

                # 临时替换无效标签为0（仅用于CRF计算）
                safe_target = torch.where(valid_mask, target, torch.zeros_like(target))

                return -self.crf_layer(logits, safe_target, mask=crf_mask, reduction="mean")
            else:
                # 展平预测和标签
                active_logits = logits.view(-1, self.class_num)
                active_labels = target.view(-1)

                # 计算交叉熵损失
                return self.loss(active_logits, active_labels)
        else:
            if self.use_crf:
                # 创建有效标签掩码（忽略填充位置）
                mask = attention_mask.bool()
                return self.crf_layer.decode(logits, mask=mask)
            else:
                # 直接返回预测 logits
                return torch.argmax(logits, dim=-1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    # 为 BERT 和其他参数设置不同学习率
    bert_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "bert" in name:
            bert_params.append(param)
        else:
            other_params.append(param)

        # 分层学习率
    params = [
        {"params": bert_params, "lr": learning_rate * config.get("bert_lr_multiplier", 0.1)},
        {"params": other_params, "lr": learning_rate}
    ]

    if optimizer == "adam":
        return Adam(params, lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(params, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"未知优化器: {optimizer}")


if __name__ == "__main__":
    from config import Config
    # 更新配置
    Config.update({
        "pretrain_model_path": "bert-base-chinese",
        "class_num": 5,  # 标签类别数
        "use_crf": True,
        "bert_lr_multiplier": 0.1,
        "dropout_rate": 0.3
    })

    # 创建模型
    model = BertCRFModel(Config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 测试输入
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(100, 2000, (batch_size, seq_len))
    print(input_ids)
    attention_mask = torch.ones((batch_size, seq_len))
    attention_mask[0, 10:] = 0  # 第一个样本后半部分为填充

    # 测试输入 - 确保标签在有效范围内
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(100, 2000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
    attention_mask[0, 10:] = 0  # 第一个样本后半部分为填充

    # 标签范围 [0, class_num-1]
    target = torch.randint(0, Config["class_num"], (batch_size, seq_len)).to(device)
    target[0, 10:] = -1  # 填充位置标签为-1

    # 验证数据
    print("验证数据...")
    print("标签范围:", torch.min(target).item(), "到", torch.max(target).item())
    print("掩码值:", torch.unique(attention_mask))

    # 测试前向传播
    print("\n测试前向传播...")
    try:
        predictions = model(input_ids, attention_mask)
        print("前向传播成功!")
        if Config["use_crf"]:
            print("CRF解码结果长度:", len(predictions))
        else:
            print("预测标签形状:", predictions.shape)
    except Exception as e:
        print("前向传播失败:", e)

    # 测试损失计算
    print("\n测试损失计算...")
    try:
        loss = model(input_ids, attention_mask, target)
        print("损失值:", loss.item())
    except Exception as e:
        print("损失计算失败:", e)