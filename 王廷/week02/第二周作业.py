import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 自定义数据集类
class RandomVectorDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = np.random.rand(num_samples, 5)  # 生成随机向量
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # 标签是最大值的索引（0-4）
        label = np.argmax(sample)
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 定义神经网络模型
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 10),  # 输入层到隐藏层
            nn.ReLU(),
            nn.Linear(10, 5)   # 隐藏层到输出层
        )
    
    def forward(self, x):
        return self.fc(x)

# 创建数据集
train_dataset = RandomVectorDataset(5000)  # 5000个训练样本
test_dataset = RandomVectorDataset(1000)   # 1000个测试样本

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每个epoch打印训练损失
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

# 测试函数
def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # 获取预测类别
            correct += pred.eq(target).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    return accuracy

# 训练5个epoch
train(epochs=5)

# 测试模型性能
test_accuracy = test()

# 示例预测
def predict_random_sample():
    model.eval()
    sample = np.random.rand(5)  # 随机生成一个五维向量
    print(f"\n随机输入向量: {sample}")
    
    with torch.no_grad():
        tensor = torch.tensor(sample, dtype=torch.float32).to(device)
        output = model(tensor)
        prob = torch.softmax(output, dim=0)  # 获取概率分布
        pred_class = output.argmax().item()
    
    true_class = np.argmax(sample)
    print(f"真实类别: {true_class} (第{true_class+1}维)")
    print(f"预测类别: {pred_class} (第{pred_class+1}维)")
    print(f"类别概率分布: {prob.cpu().numpy().round(4)}")

# 运行示例预测
predict_random_sample()
