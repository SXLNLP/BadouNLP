import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)


# 生成训练数据
def generate_data(num_samples):
    # 生成五维随机向量
    features = np.random.randn(num_samples, 5)
    # 确定每个样本的类别（最大值所在的维度）
    labels = np.argmax(features, axis=1)
    return features, labels


# 生成训练集和测试集
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义模型
class MultiClassModel(nn.Module):
    def __init__(self):
        super(MultiClassModel, self).__init__()
        # 线性层，将5维输入映射到5维输出
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        # 前向传播，计算logits
        logits = self.linear(x)
        return logits


# 初始化模型、损失函数和优化器
model = MultiClassModel()
# 使用交叉熵损失函数（已包含softmax操作）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 打印每个epoch的损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')


# 评估模型
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        # 获取预测的类别
        _, predicted = torch.max(outputs, 1)
        # 计算准确率
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')


# 训练模型
train_model(model, train_loader, criterion, optimizer)
# 评估模型
evaluate_model(model, X_test_tensor, y_test_tensor)


# 示例预测
def predict_sample(model, sample):
    model.eval()
    with torch.no_grad():
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0)
        output = model(sample_tensor)
        # 获取预测的类别和对应的概率分布
        _, predicted = torch.max(output, 1)
        # 使用softmax计算概率
        probabilities = torch.softmax(output, dim=1)
        return predicted.item(), probabilities.numpy()[0]


# 测试一个样本
sample = np.random.randn(5)
predicted_class, probs = predict_sample(model, sample)
print(f"\n示例输入: {sample}")
print(f"真实类别: {np.argmax(sample)}")
print(f"预测类别: {predicted_class}")
print(f"概率分布: {probs}")
