import torch
import torch.nn as nn
from torch.optim import Adam, SGD


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(config["vocab_size"] + 1, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)

        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # 中间层可加激活
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)  # 最后一层为纯线性
        )
        self.loss = nn.TripletMarginLoss()

    def encode(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)  # [batch, hidden*2]
        return self.projection(pooled)  # 最终输出为纯线性变换结果

    def forward(self, s1, s2=None, s3=None):
        if s2 is None and s3 is None:
            return self.encode(s1)

        e1 = self.encode(s1)
        e2 = self.encode(s2)
        e3 = self.encode(s3)
        if s2 is not None and s3 is not None:
            return self.loss(e1, e2, e3)
        return e1


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SentenceEncoder(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    s3 = torch.LongTensor([[2, 4, 6, 7], [3, 7, 8, 0]])
    y = model(s1, s2, s3)
    print(y)
