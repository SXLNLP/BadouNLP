"""
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""
import torch
import torch.nn as nn
import numpy as np

#torch.randn(参数1：样本数，参数2：维度):随机生成多少个样本中的N维向量
# 随机输入数据 (100个样本，每个样本是5维向量)
pred = torch.randn(100, 5)  # shape: [100, 5]
print(pred)

#torch.argmax:返回输入张量中所有元素的最大值的索引
# 标签：每个样本的最大值所在维度
target = torch.argmax(pred, dim=1)  # shape: [100]

#使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()

loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")

