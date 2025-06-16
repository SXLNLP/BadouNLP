import os.path
import random

import torch.nn as nn
import torch.optim

import numpy as np


# 构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
class RNNClassify(nn.Module):
    def __init__(self, vocab_size, vocab_dim, hidden_dim, out_dim):
        super(RNNClassify, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_dim)
        self.layer = nn.RNN(vocab_dim, vocab_dim, batch_first=True)
        self.classify = nn.Linear(vocab_dim, out_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding.forward(x)
        _, x = self.layer.forward(x)
        x = torch.squeeze(x)
        y_pred = self.classify(x)

        if y is None:
            return y_pred
        else:
            return y_pred, self.loss.forward(y_pred, y)


class DataSet:
    def __init__(self, str_len, sample_size, vocab: dict):
        self.str_len = str_len
        self.sample_size = sample_size
        self.vocab = vocab

    def __build_sample(self):
        items = list(self.vocab.items())
        got_a = False
        x = []
        y = []
        idx = 0
        token_position = idx
        while idx < self.str_len:
            item = random.choice(items)
            if item[1] == 0 or item[1] == 27:
                continue
            if item[0] == 'a':
                if not got_a:
                    token_position = idx
                got_a = True
            x.append(self.vocab.get(item[0]))
            idx += 1
        if not got_a:  # 没有a补一个
            i = random.randrange(0, self.str_len)
            x[i] = self.vocab.get('a')
            token_position = i

        y_ = np.zeros((self.str_len,))
        y_[token_position] = 1
        return x, y_

    def build_samples(self):

        x_list = []
        y_list = []
        for _ in range(self.sample_size):
            x, y = self.__build_sample()
            x_list.append(x)
            y_list.append(y)

        return x_list, y_list


class VocabSet:
    @staticmethod
    def get_vocab():
        vocab_dict = {}
        st = ord('a')
        ed = ord('z')
        for i in range(st, ed + 1):
            vocab_dict[chr(i)] = i - st + 1

        vocab_dict['pad'] = 0
        vocab_dict['unk'] = len(vocab_dict)
        return vocab_dict


def evaluate(model: RNNClassify, str_len, vocab):
    model.eval()

    test_size = 100

    x_, y_ = DataSet(str_len, test_size, vocab).build_samples()

    correct, wrong = 0, 0
    with torch.no_grad():
        for i in range(test_size):
            x = x_[i]
            y = y_[i]
            y_pred = model.forward(torch.LongTensor(x))

            if torch.argmax(y_pred) == np.argmax(y):
                correct += 1
            else:
                wrong += 1
    print(f"correct rate: {correct} / {correct + wrong}")


def train():
    epoch_num = 200
    sample_size = 100
    batch_size = 20
    str_len = 5
    hidden_size = 1
    vocab_dim = 12
    learning_rate = 0.001
    vocab = VocabSet().get_vocab()

    model = RNNClassify(len(vocab), vocab_dim, hidden_size, str_len)
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    model.train()

    for epoch in range(epoch_num):

        samples = DataSet(str_len, sample_size, vocab).build_samples()

        for batch_index in range(sample_size // batch_size):
            x_, y_ = samples
            x_ = x_[batch_index * batch_size:(batch_index + 1) * batch_size]
            y_ = y_[batch_index * batch_size:(batch_index + 1) * batch_size]

            optim.zero_grad()

            y_pred, loss = model.forward(torch.LongTensor(np.array(x_)), torch.FloatTensor(np.array(y_)))
            loss.backward()
            optim.step()
    evaluate(model, str_len, vocab)
    return model


def main():
    str_len = 5
    vocab = VocabSet().get_vocab()

    x_test, y_test = DataSet(str_len, 3, vocab).build_samples()
    predict(None, 'model.pth', x_test)


def predict(model: RNNClassify, path, x):
    if model is not None:
        return do_predict(model, x)
    elif os.path.exists(path):
        str_len = 5
        hidden_size = 1
        vocab_dim = 12
        vocab = VocabSet().get_vocab()
        vocab_size = len(vocab)
        model = RNNClassify(vocab_size, vocab_dim, hidden_size, str_len)
        model.load_state_dict(torch.load(path))
        return do_predict(model, x)
    else:
        model = train()
        torch.save(model.state_dict(), path)
        return do_predict(model, x)


def do_predict(model: RNNClassify, x):
    with torch.no_grad():
        x = np.array(x)
        y_pred = torch.argmax(model.forward(torch.LongTensor(np.array(x))), dim=1).numpy()
        y_true = np.argwhere(x == 1)[:, 1]
        correct_rate = np.round(np.sum(y_pred == y_true) / np.size(y_pred) * 100, 2)
        print(f"x: {x}, predict: {y_pred}, y_true: {y_true}, correct_rate: {correct_rate}")


if __name__ == '__main__':
    main()
