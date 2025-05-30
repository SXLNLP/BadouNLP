# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的网络编写
实现一个网络完成简单nlp任务
识别文本中第一个a出现的位置
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)  # embedding层
        self.linear = nn.Linear(vector_dim, 1)  # 用于每个位置输出一个得分

    def forward(self, x, y=None):
        emb = self.embedding(x)  # (batch, sen_len, vector_dim)
        scores = self.linear(emb).squeeze(-1)  # (batch, sen_len, 1) -> (batch, sen_len)
        if y is not None:
            # 交叉熵需要输入是(batch, sen_len)，标签是(batch,)
            loss = nn.functional.cross_entropy(scores, y)
            return loss
        else:
            # 预测：取分值最大的位置（即第一次出现a的位置）
            pred = torch.argmax(scores, dim=-1)
            return pred

def build_vocab():
    chars = "a你我他defghijklmnopqrstuvwxyz"  # 包含a
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab

# 生成样本：句子 & 第一个a出现的下标（或-1）
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 如果没有a，随机一个无效下标（例如sentence_length-1）
    if 'a' in x:
        label = x.index('a')
    else:
        label = sentence_length - 1  # 假设a必然在末尾
        x[-1] = 'a'  # 强行在末尾放一个a
    x_idx = [vocab.get(c, vocab["unk"]) for c in x]
    return x_idx, label

def build_dataset(sample_num, vocab, sentence_length):
    data_x, data_y = [], []
    for _ in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        data_x.append(x)
        data_y.append(y)
    return torch.LongTensor(data_x), torch.LongTensor(data_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, len(vocab))
    return model

def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    with torch.no_grad():
        pred = model(x)
        acc = (pred == y).float().mean().item()
    print(f"正确率:  {acc:.4f}")
    return acc

def main():
    # 参数
    epoch_num = 10
    batch_size = 20
    train_samples = 500
    char_dim = 20
    sentence_length = 6
    lr = 0.005

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(train_samples // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f"第{epoch + 1}轮 平均loss: {np.mean(watch_loss):.4f}")
        evaluate(model, vocab, sentence_length)

    torch.save(model.state_dict(), "model_position.pth")
    with open("vocab_position.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    x = []
    for s in input_strings:
        arr = [vocab.get(c, vocab["unk"]) for c in s]
        if len(arr) < sentence_length:
            arr += [vocab["pad"]] * (sentence_length - len(arr))
        else:
            arr = arr[:sentence_length]
        x.append(arr)

    model.eval()
    with torch.no_grad():
        pred = model(torch.LongTensor(x))
    for s, p in zip(input_strings, pred):
        print(f"输入: {s}, 第一个a位置: {p.item()}")

if __name__ == "__main__":
    main()
    test_strings = ["abcdef", "bcadea", "bbbbba", "aacdef", "xyaxyz"]
    predict("model_position.pth", "vocab_position.json", test_strings)
