import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
from torch.utils.data import Dataset, DataLoader


#  创建学习样本，生成一系列包含6个字符的字符串，并给出a第一次出现的位置，若a未出现，则返回-1

def generate_data(num_samples, seq_len=10):
    chars = string.ascii_lowercase
    X = []
    y = []
    for _ in range(num_samples):
        s = ''.join(random.choices(chars, k=seq_len))
        X.append([ord(c) - ord('a') for c in s])  # 字符转索引（0-25）
        y1 = s.find('a')  # 第一个 'a' 的位置（-1 表示没有）
        if y1 == -1:
            y1 = 10
        else:
            y1 = y1
        y.append(y1)
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# 生成训练和测试数据
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

# 自定义 Dataset
class CharDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader
training_data = DataLoader(CharDataset(X_train, y_train), batch_size=32, shuffle=True)
test_data = DataLoader(CharDataset(X_test, y_test), batch_size=32)

# 构建RNN神经网络
class RNNModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=32, output_size=11):  # 输出7类：0~10、11
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)  # 字符嵌入
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 输出7类：0~10、11

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        _, h_n = self.rnn(x)   # h_n: (1, batch_size, hidden_size)
        out = self.fc(h_n.squeeze(0))  # (batch_size, output_size)
        return out

model = RNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

train(model, training_data, epochs=100)
# 保存模型
torch.save(model, 'full_model.bin')

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    print(f"Accuracy: {correct / total * 100:.2f}%")

evaluate(model, test_data)

loaded_model = torch.load('full_model.bin')
loaded_model.eval()  # 切换到推理模式

# # 预测函数
def predict_first_a(model, s):
    char_to_idx = {chr(i): i - ord('a') for i in range(ord('a'), ord('z')+1)}
    indices = [char_to_idx[c] for c in s]
    x = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output).item()
    return pred if pred != 10 else -1

# 测试
test_str = "nfwefbreu"
print(f"预测结果: {predict_first_a(loaded_model, test_str)}")  # 输出:-1
