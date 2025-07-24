import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string

# 生成随机字符串数据集
def generate_random_string(length):
    # 生成不包含'a'的随机字符串，然后在随机位置插入'a'
    chars = string.ascii_lowercase.replace('a', '')
    s = ''.join(random.choice(chars) for _ in range(length))
    pos = random.randint(0, length-1)
    return s[:pos] + 'a' + s[pos:length-1], pos

# 创建自定义数据集类
class StringDataset(Dataset):
    def __init__(self, num_samples, string_length):
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            string, pos = generate_random_string(string_length)
            # 将字符串转换为one-hot编码
            string_tensor = torch.zeros(string_length, 26)
            for i, char in enumerate(string):
                string_tensor[i][ord(char) - ord('a')] = 1
            self.data.append(string_tensor)
            self.labels.append(pos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 定义RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 训练参数
STRING_LENGTH = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# 创建数据集和数据加载器
train_dataset = StringDataset(1000, STRING_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失函数和优化器
model = RNNClassifier(26, HIDDEN_SIZE, STRING_LENGTH)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# 测试模型
model.eval()
test_dataset = StringDataset(100, STRING_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        outputs = model(batch_data)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 示例预测
def predict_position(model, input_string):
    model.eval()
    # 将输入字符串转换为one-hot编码
    string_tensor = torch.zeros(1, STRING_LENGTH, 26)
    for i, char in enumerate(input_string[:STRING_LENGTH]):
        string_tensor[0][i][ord(char) - ord('a')] = 1
    
    with torch.no_grad():
        output = model(string_tensor)
        predicted_pos = torch.argmax(output).item()
    
    return predicted_pos

# 测试一些示例
test_strings = [
    'abcdefghij',
    'bcadefghij',
    'bcdefaghij'
]

for test_str in test_strings:
    pred_pos = predict_position(model, test_str)
    actual_pos = test_str.index('a')
    print(f'String: {test_str}')
    print(f'Predicted position: {pred_pos}')
    print(f'Actual position: {actual_pos}')
    print()
