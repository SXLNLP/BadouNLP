import numpy as np
import torch
import torch.nn as nn
import random

ASC_TRANS = 96

def max_data_in_data_to_onehot(idx, dim):
    y = np.zeros(dim)
    y[idx] = 1
    return y

####### generator ########
# 字符串中第一次出现a的位置, 为避免随机生成样本不均匀，每个样本中都会随机放置一个a
def generator_a_pos_data(dim, vocab):
    a_pos = random.randint(0, dim-1)
    x = [random.choice(list(vocab.keys())) for _ in range(dim)]
    x[a_pos] = 'a'
    y = max_data_in_data_to_onehot(a_pos, dim)
    return x, y

###### embedding ######
def build_vocab():
    vocab = {chr(i): (i - ASC_TRANS) for i in range(ord("a"), ord("z") + 1)}
    vocab["[pad]"] = 0
    vocab["[unk]"] = len(vocab)
    return vocab


def trans_str2seq(str, vocab, dim):
    seq = [vocab.get(s, vocab["[unk]"]) for s in str][0:dim]
    if len(seq) < dim:
        seq += vocab["[pad]"] * (dim - len(seq))
    return seq

#### 数据生成
def make_datas(rule, dim, nums, only_x=False):
    S = []
    X = []
    Y = []
    vocab = build_vocab()
    for idx in range(nums):
        x, y = rule(dim, vocab)
        S.append(x)
        x = trans_str2seq(x, vocab, dim)
        X.append(x)
        Y.append(y)
    """
        模型传入的是float类型，数据必须传floattensor,否则会报错
        RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
    """
    if only_x is False :
        return S, torch.FloatTensor(X), torch.FloatTensor(Y)
    else:
        return S, torch.FloatTensor(X), None

def save_data(tensor, path):
    torch.save(tensor, path)

def load_data(path):
    return torch.load(path)

if __name__ == "__main__":
    s, x, y = make_datas(generator_a_pos_data, 5, 2, only_x=False)
    print(s, x, y)