import torch
import torch.nn as nn
import numpy as np


# 生成数据
def generate_data(num_samples=10):
    data = np.random.rand(num_samples, 5)
    labels = np.argmax(data, axis=1)
    return torch.FloatTensor(data), torch.LongTensor(labels)

pred,target = generate_data(10)

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(pred, target)
print(loss.item(), "PyTorch 交叉熵")



# softmax（防溢出）
def softmax(matrix):
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

#  one-hot 编码
def to_one_hot(target, num_classes):
    one_hot = np.zeros((len(target), num_classes))
    one_hot[np.arange(len(target)), target] = 1
    return one_hot

# 手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred_softmax = softmax(pred)
    target_one_hot = to_one_hot(target, class_num)
    entropy = -np.sum(target_one_hot * np.log(pred_softmax + 1e-10), axis=1)  # 加 1e-10 防 log(0)
    return np.mean(entropy)  # 平均损失

print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")
