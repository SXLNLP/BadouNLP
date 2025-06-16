import torch
import torch.nn as nn
import numpy as np
import random
import string

"""
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
"""


# 数据生成
def generate_sample(length=5):
    # 生成一个包含'a'的随机位置
    a_position = random.randint(0, length - 1)
    chars = [random.choice(string.ascii_lowercase.replace('a', '')) for _ in range(length)]
    chars[a_position] = 'a'
    return ''.join(chars), a_position


# 编码器
class CharEncoder:
    def __init__(self, chars):
        self.all_chars = chars
        self.n_chars = len(chars)
        self.char2idx = {char: idx for idx, char in enumerate(chars)}

    def char_to_onehot(self, char):
        idx = self.char2idx[char]
        onehot = np.zeros(self.n_chars)
        onehot[idx] = 1
        return onehot

    def string_to_tensor(self, s):
        return np.array([self.char_to_onehot(c) for c in s])


# PyTorch 模型
class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out


# 参数设置
all_chars = string.ascii_lowercase
input_size = len(all_chars)
hidden_size = 4
max_length = 5
num_classes = max_length

# 初始化模型
torch_model = TorchRNN(input_size, hidden_size, num_classes)
encoder = CharEncoder(all_chars)


# 训练模型
def train_model(model, encoder, num_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        s, label = generate_sample()
        x = encoder.string_to_tensor(s)
        torch_x = torch.FloatTensor(x).unsqueeze(0)
        label_tensor = torch.tensor([label])

        optimizer.zero_grad()
        outputs = model(torch_x)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 训练模型
train_model(torch_model, encoder)

# 生成新的样本进行测试
s, label = generate_sample()
print("测试样本:", s, "实际位置:", label)

x = encoder.string_to_tensor(s)

torch_x = torch.FloatTensor(x).unsqueeze(0)
torch_out = torch_model(torch_x)
print("模型输出:", torch_out.detach().numpy())
print("模型预测位置:", torch.argmax(torch_out).item())
