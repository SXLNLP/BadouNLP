# 改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
def generate_data(num_samples):
    # 生成随机五维向量（0-1范围）
    X = np.random.rand(num_samples, 5)
    # 确定类别（最大值所在的维度，0-4）
    y = np.argmax(X, axis=1)
    return X, y

# 生成1000个样本
X, y = generate_data(1000)

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X)  # 特征张量
y_tensor = torch.LongTensor(y)   # 标签张量（必须是LongTensor）

# 2. 定义简单的神经网络模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # 单层线性模型，输入5维，输出5维（对应5个类别）
        self.layer = nn.Linear(5, 5)

    def forward(self, x):
        # 直接返回线性层输出（CrossEntropyLoss会自动加softmax）
        return self.layer(x)

# 3. 初始化模型、损失函数和优化器
model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（包含softmax）
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 简单SGD优化器

# 4. 训练模型（简化版训练循环）
print("开始训练...")
for epoch in range(100):  # 训练100轮
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每10轮打印一次进度
    if (epoch+1) % 10 == 0:
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_tensor).sum().item()
        accuracy = correct / y_tensor.size(0)

        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')

# 5. 测试模型（使用同样的数据简单演示）
with torch.no_grad():  # 不需要计算梯度
    test_outputs = model(X_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_tensor).sum().item() / y_tensor.size(0)
    print(f'\n最终准确率: {accuracy:.2f}')

# 6. 示例预测
print("\n示例预测:")
test_sample = torch.FloatTensor([[0.1, 0.5, 0.3, 0.9, 0.2]])  # 最大值在第3维(索引3)
predicted_class = model(test_sample).argmax().item()
print(f"输入向量: {test_sample.numpy()}")
print(f"预测类别: {predicted_class} (实际最大值在索引{test_sample.argmax().item()})")
