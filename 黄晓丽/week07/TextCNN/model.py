# models.py
import torch.nn as nn
import torch.nn.init as init
from config import global_config as config  # 使用全局配置
import torch


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=0
        )

        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.num_filters, (fs, config.embed_dim))
            for fs in config.filter_sizes
        ])

        # 全连接层
        self.fc = nn.Linear(
            len(config.filter_sizes) * config.num_filters,
            config.num_classes
        )

        # Dropout层
        self.dropout = nn.Dropout(config.dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 嵌入层初始化
        init.xavier_uniform_(self.embedding.weight)
        # 卷积层初始化
        for conv in self.convs:
            init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                init.constant_(conv.bias, 0.1)
        # 全连接层初始化
        init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)

        # 对每个卷积核进行卷积和池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)  # (batch_size, num_filters, seq_len - fs + 1, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, num_filters, seq_len - fs + 1)
            conv_out = nn.functional.relu(conv_out)
            pooled = nn.functional.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        # 拼接所有卷积核的输出
        x = torch.cat(conv_outputs, 1)  # (batch_size, num_filters * len(filter_sizes))

        # Dropout
        x = self.dropout(x)

        # 全连接层
        x = self.fc(x)  # (batch_size, num_classes)
        return torch.sigmoid(x)