import torch
import torch.nn as nn

# 定义交叉熵损失函数
ce_loss = nn.CrossEntropyLoss()

# 生成随机数据（3个样本，每个样本为5维随机向量）
# 随机生成预测分数（logits），形状为 (N, C) = (3, 5)
#pred = torch.randn(3, 5)  # 五分类任务，每个样本输出5个类别分数
pred = torch.FloatTensor([[1, 2, 3, 4, 5],
                          [5, 4, 3, 2, 1],
                          [4, 5, 3, 2, 1],
                          [4, 3, 5, 2, 1],
                          [4, 2, 3, 5, 1]]) #n*class_num
# 生成真实标签：取每个样本预测向量中最大值的索引作为真实类别
# 注意：这里仅为演示，实际任务中标签应为已知真实值
target = torch.argmax(pred, dim=1)  # 模拟真实标签（根据预测值最大值生成）
print("生成的真实标签（基于预测值最大值）:", target)

# 计算交叉熵损失
loss = ce_loss(pred, target)
print("交叉熵损失值:", loss.item())

# 输出每个样本的预测分数和对应真实标签
for i in range(len(pred)):
    print(f"样本{i+1}预测分数:", pred[i].numpy())
    print(f"样本{i+1}真实类别:", target[i].item())
    print("-" * 30)
