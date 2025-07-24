#-*-coding:utf-8-*-
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

HID_SIZE = 128

class HomeworkNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(HomeworkNet, self).__init__()
        self.hid_size = HID_SIZE
        self.linear = nn.Linear(input_size, self.hid_size)
        # self.linear = nn.Linear(input_size, output_size)
        # 添加隐藏层
        self.hid = nn.Linear(self.hid_size, output_size)
        self.relu = nn.LeakyReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.relu(self.linear(x))
        y_pre = self.relu(self.hid(x))
        # y_pre = x
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre

if __name__ == '__main__':
    input_size = 100
    output_size = 1
    test_model = HomeworkNet(input_size, output_size)
    test_model.train()
    test_x = torch.randn(100, input_size)
    test_y = torch.randn(100, output_size)
    test_loss = test_model(test_x, test_y)
    print("cur test_loss: {}".format(test_loss))

