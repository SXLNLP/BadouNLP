import torch
import torch.nn as nn
import torch.optim as optim
import random
import string

# 参数
SEQ_LEN = 10
VOCAB = list(string.ascii_lowercase)
VOCAB_SIZE = len(VOCAB)
BATCH_SIZE = 64
NUM_CLASSES = SEQ_LEN  # a可能出现在任意位置
EPOCHS = 100

# 字符到索引
char2idx = {c: i for i, c in enumerate(VOCAB)}

def generate_sample(seq_len=SEQ_LEN):
    pos = random.randint(0, seq_len - 1)
    chars = [random.choice(VOCAB) for _ in range(seq_len)]
    chars[pos] = 'a'
    return ''.join(chars), pos

def generate_batch(batch_size=BATCH_SIZE):
    X = []
    y = []
    for _ in range(batch_size):
        s, pos = generate_sample()
        X.append([char2idx[c] for c in s])
        y.append(pos)
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out

# 实例化模型
model = RNNClassifier(VOCAB_SIZE, embed_dim=16, hidden_dim=32, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(EPOCHS):
    X_batch, y_batch = generate_batch()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 2 == 0:
        pred = outputs.argmax(dim=1)
        acc = (pred == y_batch).float().mean().item()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

# 测试
X_test, y_test = generate_batch(10)
outputs = model(X_test)
pred = outputs.argmax(dim=1)
for i in range(10):
    s = ''.join(VOCAB[idx] for idx in X_test[i])
    print(f"字符串: {s}, 预测位置: {pred[i].item()}, 实际位置: {y_test[i].item()}")
