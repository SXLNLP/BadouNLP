import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 超参数配置
BATCH_SIZE = 64
LEARNING_RATE = 0.1
EPOCHS = 20
INPUT_DIM = 5
NUM_CLASSES = 5

# 自定义数据集生成
class RangeDataset(Dataset):
    def __init__(self, num_samples=10000):
        # 生成[1,5)范围内的均匀分布数据
        self.data = torch.rand(num_samples, INPUT_DIM) * 4 + 1
        # 生成标签（自动处理多个最大值情况）
        self.labels = torch.argmax(self.data, dim=1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 纯线性分类器
class LinearMaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, NUM_CLASSES)
        
    def forward(self, x):
        return self.linear(x)  # 直接输出logits

# 初始化组件
train_set = RangeDataset(8000)
test_set = RangeDataset(2000)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

model = LinearMaxClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计训练信息
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    
    # 验证阶段
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
    
    # 计算指标
    train_acc = 100 * correct / len(train_set)
    test_acc = 100 * test_correct / len(test_set)
    avg_loss = total_loss / len(train_set)
    
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Test Acc: {test_acc:.2f}%")

# 测试边界案例
def test_edge_cases():
    test_cases = [
        (torch.tensor([1.0, 2.0, 5.0, 3.0, 4.0]), 2),  # 明确最大值
        (torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0]), 0),  # 所有值相同
        (torch.tensor([3.0, 5.0, 5.0, 2.0, 1.0]), 1),  # 多个最大值取第一个
        (torch.tensor([1.0, 1.5, 1.0, 5.0, 1.0]), 3)  # 边界值测试
    ]
    
    model.eval()
    with torch.no_grad():
        for data, true_label in test_cases:
            output = model(data)
            pred_label = torch.argmax(output).item()
            print(f"输入: {data.numpy().round(2)} | 预测: {pred_label} | 实际: {true_label}")

print("\n边界案例测试:")
test_edge_cases()

# 查看学习到的权重矩阵
print("\n学习到的权重矩阵:")
print(model.linear.weight.detach().numpy().round(2))

