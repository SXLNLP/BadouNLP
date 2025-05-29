import torch
import torch.nn as nn
import numpy as np
import random
import json
from torch.utils.data import Dataset, DataLoader


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128, num_layers=2):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.LSTM(vector_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.classify = nn.Linear(hidden_size, sentence_length + 1)  # 输出长度为字符串长度+1（含无a的情况）
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, hidden_size)
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(last_output)  # (batch_size, sentence_length+1)

        if y is not None:
            y = y.squeeze().long()
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    """构建字符到索引的映射表，包含字母、数字、常见符号"""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


class StringDataset(Dataset):
    def __init__(self, vocab, sample_length, sentence_length):
        self.vocab = vocab
        self.sample_length = sample_length
        self.sentence_length = sentence_length

    def __len__(self):
        return self.sample_length

    def __getitem__(self, idx):
        x, y = build_sample(self.vocab, self.sentence_length)
        return torch.LongTensor(x), torch.LongTensor([y])


def build_sample(vocab, sentence_length):
    """随机生成字符串样本，并标注'a'第一次出现的位置（若无则返回0）"""
    chars = list(vocab.keys())
    chars.remove('pad')
    chars.remove('unk')

    x = [random.choice(chars) for _ in range(sentence_length)]
    position = 0
    for i, char in enumerate(x):
        if char == 'a':
            position = i + 1
            break
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, position


def build_dataset(sample_length, vocab, sentence_length):
    """构造训练/测试数据集"""
    dataset = StringDataset(vocab, sample_length, sentence_length)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def build_model(vocab, char_dim, sentence_length):
    """构建模型实例"""
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sentence_length=10):
    """评估模型准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in build_dataset(200, vocab, sentence_length):
            y_pred = model(x_batch)
            predictions = torch.argmax(y_pred, dim=1)
            correct += (predictions == y_batch.squeeze()).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    """主训练函数"""
    epoch_num = 20
    batch_size = 64
    train_sample = 3000
    char_dim = 50
    sentence_length = 10
    learning_rate = 0.001

    vocab = build_vocab()
    print("字符表:", vocab)

    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x_batch, y_batch in build_dataset(train_sample, vocab, sentence_length):
            optim.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss:{avg_loss:.4f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, avg_loss])
        torch.save(model.state_dict(), "model.pth")
        json.dump(vocab, open("vocab.json", "w", encoding="utf-8"))  # 保存词汇表

    with open("log.json", "w") as f:
        json.dump(log, f)


def predict(model_path, vocab_path, input_strings):
    """使用训练好的模型进行预测"""
    char_dim = 50
    sentence_length = 10

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = []
    for s in input_strings:
        s = s.lower()[:sentence_length].ljust(sentence_length, 'z')  # 截断或填充
        encoded = [vocab.get(char, vocab['unk']) for char in s]
        tensor_x = torch.LongTensor([encoded])
        with torch.no_grad():
            y_pred = model(tensor_x)
            pred_pos = torch.argmax(y_pred, dim=1).item()
        results.append((s, pred_pos))

    for i, (s, pos) in enumerate(results):
        print(f"输入：{s}, 预测位置：{pos}")

    return results


if __name__ == "__main__":
    # 训练模型
    main()

    # 测试预测
    test_strings = [
        "abcdefghij",      # a在位置1
        "bbacdefghi",      # a在位置3
        "xyzaxyzaxy",      # a在位置4
        "aabbccddee",      # a在位置1
        "zzzzzzzzza",      # a在位置10
        "1hello3aworld",   # 包含非字母字符,a在位置8
        "no_a_in_this",    # 没有a
        "aaaaaaaaaa",      # 多个a，应识别第一个
        "a",               # 短于sentence_length
        "xxxxxxxxxx",      # 全部无a
        "!@#a$%^&*()",     # 特殊字符中夹杂a
        "123456789a",      # 数字开头，a在末尾
    ]
    predict("model.pth", "vocab.json", test_strings)

