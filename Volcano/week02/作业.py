import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 生成训练数据
def generate_data(num_samples):
    # 生成随机五维向量
    vectors = np.random.rand(num_samples, 5)
    # 找出每个向量中最大值的索引作为标签
    labels = np.argmax(vectors, axis=1)
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(vectors)
    y = torch.LongTensor(labels)
    
    return X, y

# 定义神经网络模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(5, 10)
        self.layer2 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 训练参数设置
num_epochs = 100
batch_size = 32
learning_rate = 0.01

# 生成训练和测试数据
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

# 初始化模型、损失函数和优化器
model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 将模型设置为训练模式
    model.train()
    
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每10个epoch打印一次训练信息
    if (epoch + 1) % 10 == 0:
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

# 最终测试
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'\nFinal Test Accuracy: {accuracy:.4f}')

# 测试一些随机样本
print('\n测试随机样本示例：')
X_samples, y_samples = generate_data(5)
with torch.no_grad():
    outputs = model(X_samples)
    _, predicted = torch.max(outputs.data, 1)
    
    for i in range(len(X_samples)):
        print(f'\n输入向量: {X_samples[i].numpy()}')
        print(f'真实类别: {y_samples[i].item()}')
        print(f'预测类别: {predicted[i].item()}')
