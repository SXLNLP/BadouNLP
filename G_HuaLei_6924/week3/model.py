import json

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        embedding_dim = config['embedding_dim']
        model_type = config['model_type']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        num_classes = config['num_classes']
        with open(config['vocab_path'], 'r') as f:
            vocab = json.load(f)

        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
        if model_type == 'RNN':
            self.encoder = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        else:
            self.encoder = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.classify = nn.Linear(hidden_size, num_classes+1)
        self.activation = torch.softmax
        self.loss = nn.CrossEntropyLoss()



    def forward(self, x, y=None):
        x = self.embedding(x)
        out, _  = self.encoder(x)
        x = out[:, -1, :]
        y_pred = self.classify(x)
        if y is not None:
            # print(f'x:\n{x}')
            # print(f'y_pred:\n{y_pred}')
            # print(f'y:\n{y}')
            return self.loss(y_pred, y)
        return self.activation(y_pred, dim=-1)


def choose_optimizer(model, config):
    '''
    模型优化器选择
    :param config: 公共配置
    :param model: 优化器所属模型
    :return: 优化器
    '''

    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config import Config
    print('config:', Config)
    model = TorchModel(Config)