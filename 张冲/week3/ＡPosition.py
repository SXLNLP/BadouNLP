import json
import math
import random

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, batch_first=True,
                          bias=True)
        self.dropout = nn.Dropout1d(p=0.1)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target = None):
        encode = self.embedding(input)  # batch_size * seq_len -> batch_size * seq_len * embedding_size
        input1, hn = self.rnn(encode)  # batch_size * seq_len * embedding_size -> batch_size * hidden_size
        hn = hn.squeeze()
        input2 = self.dropout(hn)  # batch_size * hidden_size -> batch_size * hidden_size
        input3 = self.linear(input2)  # batch_size * hidden_size -> batch_size * output_size
        if target is not None:
            return self.loss(input3, target)
        else:
            return input3


def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf8') as f:
        vocab_data = json.loads(f.read())
    return vocab_data


def create_sample(seq_len, vocab):
    x = []
    for i in range(seq_len):
        x.append(random.choice(list(vocab.values())))
    if 1 in x:
        y = x.index(1)
    else:
        y = len(x)
    return x, y


def create_samples(batch_size, seq_len, vocab):
    x_list = []
    y_list = []
    for _ in range(batch_size):
        x, y = create_sample(seq_len, vocab)
        x_list.append(x)
        y_list.append(y)
    return torch.LongTensor(x_list), torch.LongTensor(y_list)


def train(vocab, model_path):
    epoch = 20
    train_data_size = 5000
    batch_size = 100
    lr = 1e-3
    seq_len = 20
    hidden_size = 128
    embedding_size = 128
    model = Model(embedding_size, len(vocab), hidden_size, seq_len + 1)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for i in range(epoch):
        loss_list = []
        for batch in range(math.floor(train_data_size / batch_size)):
            x, y = create_samples(batch_size, seq_len, vocab)
            loss = model(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_list.append(loss)
        print(f'第{i + 1}轮平均loss为:{sum(loss_list) / len(loss_list)}')
    torch.save(model.state_dict(), model_path)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(vocab, model_path):
    seq_len = 20
    hidden_size = 128
    embedding_size = 128
    model = Model(embedding_size, len(vocab), hidden_size, seq_len + 1)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())
    model.eval()
    total_cnt = 2000
    success = 0
    with torch.no_grad():
        x, y = create_samples(total_cnt, seq_len, vocab)
        y_p = model(x)
        for y_pred, y_true in zip(y_p, y):
            if np.argmax(y_pred) == y_true:
                success += 1
    print(f"正确预测个数：{success},总个数{total_cnt}, 正确率：{success * 100 / total_cnt}%")


if __name__ == '__main__':
    vocab_path = 'vocab.train'
    model_path = 'model.bin'
    vocab = load_vocab(vocab_path)
    # random.seed(42)
    # train(vocab, model_path)
    evaluate(vocab, model_path)