import numpy as np
import torch
import torch.nn as nn

def max_data_in_data_to_onehot(x, dim):
    y = np.zeros(dim)
    y[x] = 1
    return y

def generator_muti_max_data(dim):
    x = np.random.random(dim)
    y = max_data_in_data_to_onehot(np.argmax(x), dim)
    return x, y

def make_datas(rule, dim, nums, only_x=False):
    X = []
    Y = []
    for idx in range(nums):
        x, y = rule(dim)
        X.append(x)
        Y.append(y)
    """
        模型传入的是float类型，数据必须传floattensor,否则会报错
        RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
    """
    if only_x is False :
        return torch.FloatTensor(X), torch.FloatTensor(Y)
    else:
        return torch.FloatTensor(X), None

def save_data(tensor, path):
    torch.save(tensor, path)

def load_data(path):
    return torch.load(path)

if __name__ == "__main__":
    x, y = make_datas(generator_muti_max_data, 5, 2, only_x=False)
    print(x,y)