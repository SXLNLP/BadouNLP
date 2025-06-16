import torch
import torch.nn as nn
import torch.optim as optim
import random
import json

"""
基于 PyTorch 的 RNN 网络，实现多分类任务：
给定随机包含 'a' 的字符串，预测 'a' 第一次出现在字符串中的位置。
类别数等于序列长度。
"""

class RNNPositionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(RNNPositionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, _ = self.rnn(emb)   # (batch_size, seq_len, hidden_dim)
        last = out[:, -1, :]     # 取最后时刻输出 (batch_size, hidden_dim)
        logits = self.classifier(last)  # (batch_size, num_classes)
        return logits


def build_vocab():
    chars = list("abcdefghijklmnopqrstuvwxyz")
    vocab = {"pad": 0}
    for idx, c in enumerate(chars, start=1):
        vocab[c] = idx
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, seq_len):
    # 指定 'a' 第一次出现的位置
    pos = random.randint(0, seq_len - 1)
    seq = []
    for i in range(seq_len):
        if i == pos:
            seq.append('a')
        else:
            seq.append(random.choice(list(vocab.keys())))
    x = [vocab.get(ch, vocab['unk']) for ch in seq]
    return x, pos


def build_dataset(vocab, seq_len, num_samples):
    X, Y = [], []
    for _ in range(num_samples):
        x, y = build_sample(vocab, seq_len)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


def evaluate(model, vocab, seq_len, device):
    model.eval()
    X_test, Y_test = build_dataset(vocab, seq_len, num_samples=200)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == Y_test).sum().item()
    acc = correct / Y_test.size(0)
    print(f"Evaluation accuracy: {acc:.4f}")
    return acc


def predict_manual(model, vocab, seq_len, test_strings, device):
    model.eval()
    x = []
    for s in test_strings:
        seq = list(s[:seq_len])
        if len(seq) < seq_len:
            seq += ['pad'] * (seq_len - len(seq))
        x.append([vocab.get(ch, vocab['unk']) for ch in seq])
    X = torch.LongTensor(x).to(device)
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    for s, p in zip(test_strings, preds):
        print(f"输入：{s}，预测 'a' 第一次出现位置：{p}")


def main():
    # 超参数
    seq_len = 10
    embed_dim = 32
    hidden_dim = 64
    batch_size = 32
    epochs = 10
    lr = 0.001
    train_samples = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = build_vocab()
    num_classes = seq_len
    model = RNNPositionModel(len(vocab), embed_dim, hidden_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    for epoch in range(1, epochs + 1):
        model.train()
        X_train, Y_train = build_dataset(vocab, seq_len, train_samples)
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        evaluate(model, vocab, seq_len, device)

    # 保存模型与词表
    torch.save(model.state_dict(), 'rnn_position_model.pth')
    with open('vocab.json', 'w', encoding='utf8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 手动测试
    test_strings = [
        "bcdaefghij",
        "232134aaabc",
        "zzzaqqqx",
        "mnaopqrst"
    ]
    print("\n=== 手动测试 ===")
    predict_manual(model, vocab, seq_len, test_strings, device)

if __name__ == '__main__':
    main()
