import numpy as np
import torch
import torch.nn as nn
import json
import random
import matplotlib.pyplot as plt

"""
基于PyTorch的网络编写
实现一个网络完成一个简单NLP任务：
构造随机包含'a'的字符串，使用RNN进行多分类，
预测字符 'a' 第一次出现在字符串中的位置。
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size, hidden_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)  # 添加 LayerNorm
        self.dropout = nn.Dropout(0.3)        # 添加 Dropout
        self.pool = nn.AdaptiveMaxPool1d(1)   # 添加池化层
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, sentence_length)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)                      # (batch, seq_len, vector_dim)
        output, _ = self.rnn(x)                    # (batch, seq_len, hidden_dim)
        # output = self.dropout(output)              # Dropout 防过拟合
        output = self.norm(output)                 # LayerNorm over hidden_dim

        output = output.transpose(1, 2)
        pooled = self.pool(output).squeeze(-1)     # (batch, hidden_dim)

        h = self.relu(self.hidden_layer(pooled))   # 非线性变换
        logits = self.classifier(h)                # 分类输出

        if y is not None:
            return self.loss_fn(logits, y)
        else:
            return torch.argmax(logits, dim=1), torch.softmax(logits, dim=1)

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab["unk"] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    a_pos = random.randint(0, sentence_length - 1)
    chars = list(random.choices([ch for ch in vocab if ch != 'pad'], k=sentence_length))
    chars[a_pos] = 'a'
    x = [vocab.get(c, vocab['unk']) for c in chars]
    y = a_pos
    return x, y

def build_dataset(sample_num, vocab, sentence_length):
    data_x, data_y = [], []
    for _ in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        data_x.append(x)
        data_y.append(y)
    return torch.LongTensor(data_x), torch.LongTensor(data_y)

def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y_true = build_dataset(100, vocab, sentence_length)
    with torch.no_grad():
        y_pred, _ = model(x)
    correct = (y_pred == y_true).sum().item()
    acc = correct / len(y_true)
    print(f"[Evaluate] Accuracy: {acc:.2%}")
    return acc

def predict(model_path, vocab_path, input_strings):
    vector_dim = 16
    sentence_length = 6
    hidden_dim = 32

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = TorchModel(vector_dim, sentence_length, len(vocab), hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = []
    for input_string in input_strings:
        x_ids = [vocab.get(char, vocab["unk"]) for char in input_string]
        if len(x_ids) < sentence_length:
            x_ids += [vocab["pad"]] * (sentence_length - len(x_ids))
        else:
            x_ids = x_ids[:sentence_length]
        x.append(x_ids)

    x_tensor = torch.LongTensor(x)
    with torch.no_grad():
        preds, probs = model(x_tensor)

    for i, input_string in enumerate(input_strings):
        prob = probs[i].tolist()
        pred_class = preds[i].item()
        one_hot = [1 if j == pred_class else 0 for j in range(len(prob))]

        print(f"输入：{input_string}")
        print(f"预测位置概率分布：{[round(p, 4) for p in prob]}")
        print(f"对应的 one-hot 向量：{one_hot}")
        print(f"==> 该输入被判为第 {pred_class} 类\n")

def main():
    epoch_num = 20
    batch_size = 32
    sentence_length = 6
    vector_dim = 16
    hidden_dim = 32

    vocab = build_vocab()
    model = TorchModel(vector_dim, sentence_length, len(vocab), hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    log = []

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for _ in range(100):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / 100
        print(f"[Train] Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, avg_loss])  # 记录acc和loss

    torch.save(model.state_dict(), "rnn_model.pth")
    with open("rnn_vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("\n[Test Predict]")
    test_strings = ["azzzzz", "zazzzz", "zzazzz", "zzzzaa"]
    predict("rnn_model.pth", "rnn_vocab.json", test_strings)

    draw_train_log(log)

def draw_train_log(log):
    acc = [item[0] for item in log]
    loss = [item[1] for item in log]
    epochs = list(range(1, len(log) + 1))

    plt.figure(figsize=(10, 4))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, marker='o')
    plt.title("Evaluation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
