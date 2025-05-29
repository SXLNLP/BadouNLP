import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

# 参数设置
STRING_LENGTH = 5  # 字符串长度
VOCAB_SIZE = 26  # 字母表大小（a-z）
NUM_CLASSES = STRING_LENGTH + 1  # 类别数（位置0-5，5表示没有'a'）
HIDDEN_SIZE = 32  # 隐藏层大小
BATCH_SIZE = 64   # 批量大小
EPOCHS = 20   # 训练轮数


# 生成随机字符串和标签
def generate_string_and_label():
    chars = [chr(ord('a') + i) for i in range(VOCAB_SIZE)]
    s = ''.join(random.choices(chars, k=STRING_LENGTH))
    label = s.find('a')  # 找到第一个'a'的位置
    if label == -1:
        label = STRING_LENGTH  # 如果没有'a'，标记为STRING_LENGTH
    return s, label


# 将字符串转换为one-hot编码
def string_to_tensor(s):
    tensor = torch.zeros(len(s), VOCAB_SIZE)
    for i, ch in enumerate(s):
        tensor[i][ord(ch) - ord('a')] = 1
    return tensor


# 自定义数据集
class StringDataset(Dataset):
    def __init__(self, num_samples):
        self.data = []
        for _ in range(num_samples):
            s, label = generate_string_and_label()
            self.data.append((s, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, label = self.data[idx]
        x = string_to_tensor(s)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# RNN模型
class PositionPredictor(nn.Module):
    def __init__(self):
        super(PositionPredictor, self).__init__()
        self.rnn = nn.RNN(
            input_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)  # out shape: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out


# 训练函数
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


# 测试函数
def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


# 主程序
if __name__ == "__main__":
    # 准备数据
    train_dataset = StringDataset(5000)
    test_dataset = StringDataset(1000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    model = PositionPredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和测试
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, criterion)
        print(f'第 {epoch + 1}/{EPOCHS} 轮训练: '
              f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | '
              f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')


    # 示例预测
    def predict_position(model, s):
        model.eval()
        x = string_to_tensor(s).unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            pos = predicted.item()
            return pos if pos != STRING_LENGTH else -1


    test_strings = ["bcdea", "xyzab", "bbbbb", "aabaa"]
    print("\n测试字符串预测结果:")
    correct_predictions = 0
    for s in test_strings:
        pred_pos = predict_position(model, s)
        true_pos = s.find('a')
        is_correct = pred_pos == true_pos
        if is_correct:
            correct_predictions += 1
        print(f"字符串: '{s}' -> 预测'a'首次位置: {pred_pos} (真实位置: {true_pos}) {'✓' if is_correct else '✗'}")

    # 计算并打印预测准确率
    prediction_accuracy = 100 * correct_predictions / len(test_strings)
    print(f"\n预测准确率: {prediction_accuracy:.1f}% ({correct_predictions}/{len(test_strings)})")
