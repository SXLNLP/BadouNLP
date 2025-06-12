#-*-coding:utf-8-*-
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

HID_SIZE = 128

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x,_ = self.rnn(x)
        y_pre = self.linear(x)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre

if __name__ == '__main__':
    input_size = 100
    hidden_size = 128
    output_size = 1
    test_model = RNNNet(input_size, hidden_size, output_size)
    test_model.train()
    test_x = torch.randn(100, input_size)
    test_y = torch.randn(100, output_size)
    test_loss = test_model(test_x, test_y)
    print("cur test_loss: {}".format(test_loss))

