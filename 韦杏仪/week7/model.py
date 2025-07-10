import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config  # 导入配置


# RNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super().__init__()
        # 词嵌入层：将词索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN层：处理序列数据
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        # 全连接层：将RNN的输出转换为分类结果
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)  # 防止过拟合

    def forward(self, input_ids):
        # input_ids形状: [batch_size, seq_length]
        embedded = self.dropout(self.embedding(input_ids))  # 嵌入层输出: [batch, seq, embed]
        # RNN输出: output[batch, seq, hidden], hidden[n_layers, batch, hidden]
        output, hidden = self.rnn(embedded)
        # 只使用最后一个时间步的隐藏状态
        # hidden[-1]是最后一层的隐藏状态
        return self.fc(self.dropout(hidden[-1]))  # 输出: [batch, output_dim]

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：比RNN多了细胞状态，更擅长处理长距离依赖
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))

        # LSTM输出: output[batch, seq, hidden], (hidden, cell)[n_layers, batch, hidden]
        output, (hidden, cell) = self.lstm(embedded)

        # 使用最后一个时间步的隐藏状态
        return self.fc(self.dropout(hidden[-1]))

# TextCNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

# Gated CNN模型
class GatedCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, kernel_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.gate1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.gate2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)

        # 第一个门控卷积层
        conv1 = self.conv1(embedded)
        gate1 = torch.sigmoid(self.gate1(embedded))
        x = conv1 * gate1

        # 第二个门控卷积层
        conv2 = self.conv2(x)
        gate2 = torch.sigmoid(self.gate2(x))
        x = conv2 * gate2

        # 全局平均池化
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)

        return self.fc(x)

def get_model(model_type, vocab_size):
    output_dim = 2

    if model_type == "rnn":
        return RNNModel(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=128,
            output_dim=output_dim,
            n_layers=2,
            dropout=0.5
        )
    elif model_type == "lstm":
        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=128,
            output_dim=output_dim,
            n_layers=2,
            dropout=0.5
        )
    elif model_type == "textcnn":
        return TextCNN(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            output_dim=output_dim,
            dropout=0.5
        )
    elif model_type == "gated_cnn":
        return GatedCNN(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=128,
            output_dim=output_dim,
            kernel_size=3,
            dropout=0.5
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
