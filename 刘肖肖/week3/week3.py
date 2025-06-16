import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 生成随机字符串及其标签
def generate_data(num_samples, seq_length, vocab_size=26):
    X = []
    y = []

    for _ in range(num_samples):
        # 确保至少有一个'a'
        a_pos = random.randint(0, seq_length - 1)
        seq = [random.randint(0, vocab_size - 1) for _ in range(seq_length)]
        seq[a_pos] = 0  # 'a'的索引为0

        X.append(seq)
        y.append(a_pos)

    return np.array(X), np.array(y)


# 将字符串转换为one-hot编码
def one_hot_encode(sequences, vocab_size):
    encoded = np.zeros((len(sequences), len(sequences[0]), vocab_size))
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            encoded[i, t, char] = 1
    return encoded


# RNN模型定义
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 使用GRU作为RNN单元
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层用于分类
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播RNN
        out, _ = self.rnn(x, h0)

        # 我们使用最后一个时间步的输出进行分类
        out = self.fc(out[:, -1, :])
        return out


# 训练模型
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')


# 评估模型
def evaluate_model(model, test_loader, device):
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

    print(f'Accuracy: {100 * correct / total:.2f}%')
    return correct / total


# 主函数
def main():
    # 配置参数
    seq_length = 10  # 字符串长度
    vocab_size = 26  # 字母表大小
    hidden_size = 64  # RNN隐藏层大小
    num_classes = seq_length  # 分类类别数（a首次出现的位置）
    num_samples = 10000  # 生成的样本数
    batch_size = 64
    learning_rate = 0.001
    epochs = 15

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成数据
    X, y = generate_data(num_samples, seq_length)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为one-hot编码
    X_train_oh = one_hot_encode(X_train, vocab_size)
    X_test_oh = one_hot_encode(X_test, vocab_size)

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_oh)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_oh)
    y_test_tensor = torch.LongTensor(y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型
    model = RNNClassifier(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        num_layers=2,
        dropout=0.2
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print("开始训练模型...")
    train_model(model, train_loader, criterion, optimizer, device, epochs)

    # 评估模型
    print("\n在测试集上评估模型:")
    accuracy = evaluate_model(model, test_loader, device)

    # 示例预测
    print("\n示例预测:")
    sample_idx = random.randint(0, len(X_test) - 1)
    sample_seq = X_test[sample_idx]
    sample_input = X_test_tensor[sample_idx].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(sample_input)
        _, predicted = torch.max(output, 1)

    # 将索引转换为字符串
    sample_str = ''.join([chr(ord('a') + idx) for idx in sample_seq])
    true_pos = y_test[sample_idx]
    pred_pos = predicted.item()

    print(f"字符串: {sample_str}")
    print(f"真实位置: {true_pos}")
    print(f"预测位置: {pred_pos}")
    print(f"预测正确: {true_pos == pred_pos}")


if __name__ == "__main__":
    main()
