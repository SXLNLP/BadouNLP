import torch
import torch.nn as nn
import torch.optim as optim

# 数据准备
num_samples = 10000
X = torch.rand(num_samples, 5)  # 生成输入数据
y = torch.argmax(X, dim=1)  # 标签为最大值所在的维度


# 定义神经网络模型
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 128)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 5)  # 隐藏层到输出层

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = ClassificationModel()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
epochs = 50
for epoch in range(epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_data = torch.rand(100, 5)  # 生成测试数据
    true_labels = torch.argmax(test_data, dim=1)
    pred_logits = model(test_data)
    pred_labels = torch.argmax(pred_logits, dim=1)

    accuracy = (pred_labels == true_labels).float().mean()
    print(f"测试准确率: {accuracy.item() * 100:.2f}%")
