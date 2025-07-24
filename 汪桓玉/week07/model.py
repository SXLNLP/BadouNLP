# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel



class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        max_length = config["max_length"]

        self.use_bert = False
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        if model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy

        # 提前创建池化层
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(max_length)
        else:
            self.pooling_layer = nn.AvgPool1d(max_length)

    def forward(self, x, target=None):
        # 初始输入
        # x: [batch_size, seq_len]  例如：[32, 30]
        
        if self.use_bert:
            # BERT分支
            if isinstance(self.encoder, BertModel):
                outputs = self.encoder(x)
                # outputs: 元组 (sequence_output, pooler_output)
                # outputs[0]: [batch_size, seq_len, 768]  例如：[32, 30, 768]
                # outputs[1]: [batch_size, 768]          例如：[32, 768]
                
                if isinstance(outputs, tuple):
                    x = outputs[0]  # 取序列输出
                    # x: [batch_size, seq_len, 768]  例如：[32, 30, 768]
            else:
                x = self.encoder(x)
                # x: [batch_size, seq_len, hidden_size]
        else:
            # 非BERT分支
            x = self.embedding(x)
            # x: [batch_size, seq_len, hidden_size]  例如：[32, 30, 128]
            
            x = self.encoder(x)
            # RNN输出: (output, hidden)
            # LSTM输出: (output, (hidden, cell))
            # GatedCNN输出: 直接张量
            
            # RNN/LSTM情况：
            # x: 元组，x[0] = [batch_size, seq_len, hidden_size]  例如：[32, 30, 128]
            # GatedCNN情况：
            # x: [batch_size, seq_len, hidden_size]  例如：[32, 30, 128]

        # 处理RNN/LSTM的输出元组
        if isinstance(x, tuple):
            x = x[0]  # 取序列输出
            # x: [batch_size, seq_len, hidden_size]  例如：[32, 30, 128]或[32, 30, 768]

        # 池化层处理
        if x.dim() == 3:
            # 确认此时 x: [batch_size, seq_len, hidden_size]
            
            x = x.transpose(1, 2)  # 转置维度
            # x: [batch_size, hidden_size, seq_len]  例如：[32, 128, 30]或[32, 768, 30]
            
            x = self.pooling_layer(x)  # 应用池化
            # MaxPool1d(max_length) 或 AvgPool1d(max_length)
            # x: [batch_size, hidden_size, 1]  例如：[32, 128, 1]或[32, 768, 1]
            
            x = x.squeeze(-1)  # 移除最后一个维度
            # x: [batch_size, hidden_size]  例如：[32, 128]或[32, 768]

        predict = self.classify(x)
        # predict: [batch_size, class_num]  例如：[32, 2]

        if target is not None:
            return self.loss(predict, target.squeeze())
            # 返回标量损失值
        else:
            return predict
            # 返回预测结果: [batch_size, class_num]


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):
        # 输入尺寸: [batch, seq_len, hidden_size]
        # 转置为: [batch, hidden_size, seq_len]
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # 转置回: [batch, seq_len, hidden_size]
        return x.transpose(1, 2)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


def choose_optimizer(config, model):
    """优化器选择函数"""
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")