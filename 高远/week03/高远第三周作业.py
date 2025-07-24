import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string

# 设置随机种子确保结果可复现
torch.manual_seed(42)
random.seed(42)


# 生成随机字符串及对应标签
def generate_sample(max_length=10):
    # 字符串长度为2到max_length之间的随机值
    length = random.randint(2, max_length)
    # 随机选择'a'首次出现的位置(0到length-1)
    a_position = random.randint(0, length - 1)

    # 生成不包含'a'的字符集
    other_chars = [c for c in string.ascii_lowercase if c != 'a']

    # 构造字符串
    s = []
    for i in range(length):
        if i == a_position:
            s.append('a')
        else:
            s.append(random.choice(other_chars))

    return ''.join(s), a_position


# 字符编码为one-hot向量
def char_to_onehot(char, vocab_size=26):
    # a:0, b:1, ..., z:25
    idx = ord(char) - ord('a')
    onehot = torch.zeros(vocab_size)
    onehot[idx] = 1
    return onehot


# 字符串编码为one-hot矩阵
def string_to_onehot(s, max_length=10, vocab_size=26):
    # 字符串转one-hot矩阵，不足max_length则填充零向量
    matrix = torch.zeros(max_length, vocab_size)
    for i, char in enumerate(s):
        matrix[i] = char_to_onehot(char)
    return matrix


# RNN模型定义
class RNNClassifier(nn.Module):
    def __init__(self, input_size=26, hidden_size=64, output_size=10):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # 初始化隐藏状态
        # 前向传播RNN
        out, _ = self.rnn(x)
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 训练函数
def train_model(model, criterion, optimizer, epochs=1000, max_length=10):
    model.train()
    for epoch in range(epochs):
        # 生成一批训练数据
        batch_size = 32
        inputs = torch.zeros(batch_size, max_length, 26)  # 26个字母的one-hot
        targets = torch.zeros(batch_size, dtype=torch.long)

        for i in range(batch_size):
            s, label = generate_sample(max_length)
            inputs[i] = string_to_onehot(s, max_length)
            targets[i] = label

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 测试函数
def test_model(model, max_length=10, num_tests=100):
    model.eval()
    correct = 0

    with torch.no_grad():
        for _ in range(num_tests):
            s, label = generate_sample(max_length)
            input_tensor = string_to_onehot(s, max_length).unsqueeze(0)  # 添加batch维度

            # 模型预测
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

            if predicted.item() == label:
                correct += 1
            else:
                print(f"预测错误: 字符串 '{s}', 实际位置: {label}, 预测位置: {predicted.item()}")

    accuracy = correct / num_tests
    print(f'测试准确率: {accuracy * 100:.2f}%')


# 初始化模型、损失函数和优化器
model = RNNClassifier(input_size=26, hidden_size=64, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和测试模型
print("开始训练模型...")
train_model(model, criterion, optimizer)
print("\n测试模型:")
test_model(model)


# 示例预测
def predict_example(model, s, max_length=10):
    model.eval()
    with torch.no_grad():
        input_tensor = string_to_onehot(s, max_length).unsqueeze(0)
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        print(f"字符串: '{s}'")
        print(f"预测'a'首次出现位置: {predicted.item()}")
        print(f"实际'a'首次出现位置: {s.index('a')}")


# 测试几个示例
print("\n示例预测:")
predict_example(model, "bcdaef")
predict_example(model, "aab")
predict_example(model, "xyzabc")


# 输出结果
# 开始训练模型...
# Epoch [100/1000], Loss: 2.0060
# Epoch [200/1000], Loss: 1.3382
# Epoch [300/1000], Loss: 0.3084
# Epoch [400/1000], Loss: 0.1770
# Epoch [500/1000], Loss: 0.0891
# Epoch [600/1000], Loss: 0.0329
# Epoch [700/1000], Loss: 0.0282
# Epoch [800/1000], Loss: 0.0192
# Epoch [900/1000], Loss: 0.0162
# Epoch [1000/1000], Loss: 0.0122
#
# 测试模型:
# 测试准确率: 100.00%
#
# 示例预测:
# 字符串: 'bcdaef'
# 预测'a'首次出现位置: 3
# 实际'a'首次出现位置: 3
# 字符串: 'aab'
# 预测'a'首次出现位置: 0
# 实际'a'首次出现位置: 0
# 字符串: 'xyzabc'
# 预测'a'首次出现位置: 3
# 实际'a'首次出现位置: 3
