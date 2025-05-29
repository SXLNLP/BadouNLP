import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import json


# 构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

class RNN_TorchDemo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(RNN_TorchDemo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # embedding(整个词汇表的size，每个词的size)
        self.norm = nn.LayerNorm(embedding_dim)                                  # norm(每个词的size)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)           # rnn(每个词的size，隐藏维度)
        self.dropout = nn.Dropout(0.3)                                           # dropout，防止过拟合
        self.fc = nn.Linear(hidden_dim, output_size)                             # 线性层(向量大小，输出类别之类)
        self.loss = nn.CrossEntropyLoss()                                        # 损失函数()

    def forward(self, x, y=None):
        embedded = self.embedding(x)
        embedded = self.norm(embedded)          # 如果没有batch_first=True，则需要.permute(1, 0, 2)

        rnn_out, rnn_hidden = self.rnn(embedded)
        last_hidden = rnn_out[:, -1, :]

        out = self.dropout(last_hidden)
        logits = self.fc(out)

        if y is not None:
            return logits, self.loss(logits, y)
        else:
            return logits


# 构建词表
def build_vocab():
    vocab = {"<PAD>": 0, "<UNK>": 1}
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    vocab.update({ch: idx for idx, ch in enumerate(chars, start=2)})
    # print(vocab)
    return vocab


# 构建训练数据，变成向量
def build_dataset(sample_num, max_len, vocab):
    # 随机生成字符，长度不能超过6,不足补零，超出截断
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    x_data, y_data = [], []
    for i in range(sample_num):
        temp_x = [random.choice(chars) for j in range(max_len)]
        temp_y = temp_x.index("a") if "a" in temp_x else max_len
        x_data.append([vocab.get(x, vocab["<UNK>"]) for x in temp_x])
        y_data.append(temp_y)

    return torch.LongTensor(x_data), torch.LongTensor(y_data)


# 训练
def train_model(model, dataloader, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        loss_list = []                      # 计算按平均loss用
        total_correct = 0                   # 统计预测正确的个数
        for x_data, y_data in dataloader:
            optimizer.zero_grad()
            logits, loss = model(x_data, y_data)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            y_pred = torch.argmax(logits, dim=-1)
            total_correct += (y_pred == y_data).sum().item()
        print(f"预测第{epoch + 1}轮数：正确个数为{total_correct}个，错误个数为{len(dataloader.dataset) - total_correct}个，平均Loss为{np.mean(loss_list)}")


def predict_samples(x_data, model_path, embedding_dim, hidden_dim):
    # 加载词表
    with open("week3_RNNDemo_work_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # 重建模型
    vocab_size = len(vocab)
    model = RNN_TorchDemo(vocab_size, embedding_dim, hidden_dim, 7)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 处理字符串  ["abcdes","kjhfaa"]
    input_ids = []
    for x in x_data:
        chars = list(x)  # 直接对字符串进行分解
        ids = []
        for ch in chars:
            ids.append(vocab[ch]) if ch in vocab else ids.append(vocab["<UNK>"])
        # 补齐或截断到 max_len 长度
        if len(ids) < 6:
            ids += [vocab.get("<PAD>", 0)] * (6 - len(ids))
        else:
            ids = ids[:6]
        input_ids.append(ids)
    x_tensor = torch.LongTensor(input_ids)

    # 预测
    with torch.no_grad():
        logits = model(x_tensor)  # forward时y传None，返回logits
        y_pred = torch.argmax(logits, dim=-1)
        math_p = torch.softmax(logits, dim=-1)
        for i in range(len(x_tensor)):
            input_ids = x_tensor[i].tolist()
            pred_class = y_pred[i].item()
            pred_prob = math_p[i][pred_class].item()
            print(f"输入的字符串：{x_data[i]}，输入的字符串编码：{input_ids}，预测类别为：{pred_class}，概率值：{pred_prob:.4%}")


def main():
    vocab = build_vocab()       # 词表
    vocab_size = len(vocab)     # 词表大小
    embedding_dim = 8           # 每个词的向量大小
    hidden_dim = 32             # 隐藏层向量大小
    max_len = 6                 # 每个句子最大长度为6
    output_size = 7             # 7类别，不包含a的为第7类
    sample_num = 10000            # 总样本数
    batch_size = 20             # 每轮每次样本数为20，如：共5次弄完100样本
    epochs = 10                 # 训练10轮
    lr = 0.01

    x_data, y_data = build_dataset(sample_num, max_len, vocab=vocab)
    dataloader = DataLoader(TensorDataset(x_data, y_data), batch_size, shuffle=True)

    model = RNN_TorchDemo(vocab_size, embedding_dim, hidden_dim, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    train_model(model, dataloader, optimizer, epochs)

    # 保存模型
    torch.save(model.state_dict(), "week3_RNNDemo_work_model.pth")
    with open("week3_RNNDemo_work_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print("模型和词表已保存！")


if __name__ == '__main__':
    main()
    x_data = ["bsafsw", "easfsw", "poiuyq", "zxcvaf", "kmjuqa"]
    predict_samples(x_data, model_path="week3_RNNDemo_work_model.pth", embedding_dim=8, hidden_dim=32)
