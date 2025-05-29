"""
Week3 作业：
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# 词表
vocab = ['<pad>', '<unk>', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
vocab_size = len(vocab)
char_to_idx = {c: i for i, c in enumerate(vocab)}
pad_idx = 0 
unk_idx = 1 

def generate_data(num_samples, include_unk=False):
    """生成长度5随机字符串，每个字符串包含至少一个'a'，标签为'a'首次出现的位置"""
    seq_length = 5

    inputs = np.zeros((num_samples, seq_length), dtype=np.int64)
    labels = np.zeros(num_samples, dtype=np.int64)
    
    base_chars = list(range(3, vocab_size))
    
    for i in range(num_samples):
        first_a_pos = np.random.randint(1, seq_length + 1)
        labels[i] = first_a_pos - 1
        char_pool = base_chars
        if include_unk:
            char_pool += [unk_idx]
        for j in range(first_a_pos - 1):
            inputs[i, j] = np.random.choice(char_pool)
        inputs[i, first_a_pos - 1] = char_to_idx['a']
        for j in range(first_a_pos, seq_length):
            inputs[i, j] = np.random.randint(0, vocab_size)
    
    return torch.from_numpy(inputs), torch.from_numpy(labels)

# RNN
class ModelRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(ModelRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader, device):
    """ 评估模型 """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

def predict(model, input_str, char_to_idx, seq_length, device):
    """预测字符串中'a'首次出现的位置"""
    indices = [char_to_idx.get(c, unk_idx) for c in input_str]
    # Padding
    if len(indices) < seq_length:
        indices += [pad_idx] * (seq_length - len(indices))
    else:
        indices = indices[:seq_length]
    
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item() + 1

def main():
    # 超参数方便修改
    seq_length = 5
    embed_dim = 16 
    hidden_dim = 32 
    output_dim = seq_length
    batch_size = 64
    epochs = 15
    
    train_inputs, train_labels = generate_data(5000, include_unk=True)
    test_inputs, test_labels = generate_data(1000, include_unk=True)
    
    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # GPU: RTX3070Ti
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ModelRNN(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    train_model(model, train_loader, criterion, optimizer, device, epochs)
    
    evaluate_model(model, test_loader, device)
    
    # 保存
    torch.save(model.state_dict(), 'rnn_position_model.pth')
    
    test_input = [
        "abcde",
        "bcdae", 
        "bcdea",
        "bbbab", 
        "xabbb",
        "ab#$%",
    ]
    
    print("\n测试预测结果:")
    for s in test_input:
        pos = predict(model, s, char_to_idx, seq_length, device)
        actual_pos = s.find('a') + 1 if 'a' in s else -1
        print(f"{s} : 预测分类: {pos} ; 实际类别: {actual_pos}")

if __name__ == "__main__":
    main()
