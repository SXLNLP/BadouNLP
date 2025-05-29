import torch
import torch.nn as nn
import numpy as np
import random

# 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 构造数据集
def generate_data(num_samples, max_length, vocab_size):
    X = []
    y = []
    for _ in range(num_samples):
        length = random.randint(1, max_length)
        seq = [random.randint(0, vocab_size - 1) for _ in range(length)]
        if 0 not in seq:  # 确保'a'在序列中
            idx_a = random.randint(0, length - 1)
            seq[idx_a] = 0  # 假设0代表'a'
        else:
            idx_a = seq.index(0)
        X.append(seq)
        y.append(idx_a)
    return X, y

# 将数据转换为Tensor
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).unsqueeze(1)

# 参数设置
num_samples = 1000
max_length = 20
vocab_size = 4  # 假设有4个不同的字符
embedding_dim = 5
hidden_size = 10
output_size = max_length  # 输出是第一个'a'的位置

# 生成数据
X, y = generate_data(num_samples, max_length, vocab_size)

# 创建词汇表
vocab = ['a', 'b', 'c', 'd']
word_to_ix = {word: i for i, word in enumerate(vocab)}

# 准备数据
inputs = [prepare_sequence(seq, word_to_ix) for seq in X]
targets = torch.tensor(y, dtype=torch.long)

# 初始化模型、损失函数和优化器
model = SimpleRNN(embedding_dim, hidden_size, output_size)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(30):  # 多次迭代
    total_loss = 0
    for i in range(len(inputs)):
        model.zero_grad()

        input_tensor = inputs[i].float()  # 转换为浮点数
        target_tensor = targets[i]

        # 前向传播
        output = model(input_tensor)
        loss = loss_function(output, target_tensor.unsqueeze(0))

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(inputs)}")

# 测试模型
test_seq = ['b', 'd', 'a', 'c']  # 示例测试序列
test_input = prepare_sequence(test_seq, word_to_ix).float()
with torch.no_grad():
    test_output = model(test_input)
    _, predicted_idx = torch.max(test_output.data, 1)
    print(f"Test sequence: {''.join(test_seq)}")
    print(f"Predicted index of first 'a': {predicted_idx.item()}")


