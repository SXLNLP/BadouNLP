# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig  # 导入BERT相关模块

"""
建立基于BERT的网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        # 获取配置参数
        self.bert_path = config.get("bert_path", "bert-base-chinese")  # BERT预训练模型路径
        self.finetune = config.get("finetune", True)  # 是否微调BERT
        class_num = config["class_num"]  # 标签类别数量
        self.use_crf = config["use_crf"]  # 是否使用CRF
        
        # 初始化BERT模型
        self.bert = BertModel.from_pretrained(self.bert_path)
        
        # 如果不微调BERT，则冻结其参数
        if not self.finetune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 获取BERT的隐藏层大小
        bert_config = BertConfig.from_pretrained(self.bert_path)
        hidden_size = bert_config.hidden_size
        
        # 分类层（将BERT输出映射到标签空间）
        self.classify = nn.Linear(hidden_size, class_num)
        
        # CRF层
        if self.use_crf:
            self.crf_layer = CRF(class_num, batch_first=True)
        
        # 交叉熵损失函数（用于非CRF情况）
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略标签为-1的位置

    def forward(self, input_ids, attention_mask, target=None):
        # 通过BERT模型
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取最后一层的隐藏状态 [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        
        # 分类层预测
        predict = self.classify(sequence_output)  # [batch_size, seq_len, class_num]
        
        if target is not None:
            if self.use_crf:
                # 创建mask（忽略padding位置）
                mask = attention_mask.bool()
                # CRF损失（负对数似然）
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 计算交叉熵损失
                # 调整形状: (batch_size * seq_len, class_num) 和 (batch_size * seq_len)
                active_loss = attention_mask.view(-1) == 1
                active_logits = predict.view(-1, predict.size(-1))[active_loss]
                active_labels = target.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
        else:
            if self.use_crf:
                # 使用CRF进行解码
                mask = attention_mask.bool()
                return self.crf_layer.decode(predict, mask)
            else:
                # 直接返回预测结果
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    # 区分BERT参数和其他参数（如果微调）
    if model.finetune:
        # 微调BERT时，使用不同的学习率
        bert_params = list(model.bert.named_parameters())
        classifier_params = list(model.classify.named_parameters())
        if model.use_crf:
            classifier_params += list(model.crf_layer.named_parameters())
        
        # 设置不同的优化器组
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_params], "lr": learning_rate * 0.1},
            {"params": [p for n, p in classifier_params], "lr": learning_rate}
        ]
        
        if optimizer == "adam":
            return Adam(optimizer_grouped_parameters)
        elif optimizer == "sgd":
            return SGD(optimizer_grouped_parameters)
    else:
        # 不微调BERT时，只优化分类层
        params_to_optimize = list(model.classify.parameters())
        if model.use_crf:
            params_to_optimize += list(model.crf_layer.parameters())
        
        if optimizer == "adam":
            return Adam(params_to_optimize, lr=learning_rate)
        elif optimizer == "sgd":
            return SGD(params_to_optimize, lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    
    # 修改配置以包含BERT相关参数
    Config["bert_path"] = "bert-base-chinese"  # 使用中文BERT模型
    Config["finetune"] = True  # 是否微调BERT
    Config["use_crf"] = True  # 是否使用CRF
    
    # 注意：输入需要调整为BERT的输入格式
    # 示例输入 (实际使用时需要tokenizer处理)
    input_ids = torch.tensor([[101, 2345, 3344, 567, 102]])  # [CLS] ... [SEP]
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])  # 注意力掩码
    targets = torch.tensor([[-1, 0, 1, 2, -1]])  # 标签（-1表示忽略）
    
    model = TorchModel(Config)
    
    # 测试前向传播
    loss = model(input_ids, attention_mask, targets)
    print("Loss:", loss.item())
    
    # 测试预测
    predictions = model(input_ids, attention_mask)
    print("Predictions:", predictions)
