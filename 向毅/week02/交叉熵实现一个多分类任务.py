import numpy as np
import torch
import torch.nn as nn

'''
手动实现交叉熵的计算
'''
# 改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

# 使用torch计算交叉熵
# arr = np.random.random(15).reshape(3, 5)
# pred = torch.FloatTensor(arr)

pred = torch.FloatTensor([
    [0.6589, 0.2852, 0.1755, 0.5153, 0.4722],
    [0.4248, 0.2110, 0.9493, 0.8906, 0.1262],
    [0.1202, 0.4634, 0.2176, 0.3713, 0.9309]])
target = torch.LongTensor([0, 2, 4])
loss = nn.CrossEntropyLoss(pred, target)
print(loss, "torch输出交叉熵")


# 实现softmax函数
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)


# 将输入转化为onehot矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target


# 手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis=1)
    return sum(entropy) / batch_size


print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")
