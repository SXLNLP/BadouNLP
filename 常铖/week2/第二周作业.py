import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义类别标签（汉字显示）
categories = ['第一类', '第二类', '第三类', '第四类', '第五类']

# 生成模拟数据
n_samples = 60
X = torch.randn(n_samples, 5)
y = torch.argmax(X, dim=1)  # 原始数字标签

# 划分训练集和测试集
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 定义神经网络模型
class MaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(x)


# 初始化训练组件
model = MaxClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练循环
epochs = 20
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 验证准确率
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predicted = torch.argmax(test_outputs, dim=1)
        accuracy = (predicted == y_test).float().mean()
        print(f'Epoch {epoch + 1}/{epochs}, 测试准确率: {accuracy:.4f}')

# 最终测试与中文显示
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    final_pred = torch.argmax(test_outputs, dim=1)
    final_acc = (final_pred == y_test).float().mean()

    # 转换为汉字标签
    y_test_cn = [categories[i] for i in y_test.numpy()]
    pred_cn = [categories[i] for i in final_pred.numpy()]

    print(f'\n最终测试准确率: {final_acc:.4f}')
    print("\n前10个测试样本预测结果：")
    for i in range(10):
        print(
            f"样本{i + 1} 实际值：{y_test_cn[i]:<5} 预测值：{pred_cn[i]:<5} {'✅' if y_test[i] == final_pred[i] else '❌'}")
