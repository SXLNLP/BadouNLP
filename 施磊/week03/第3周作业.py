#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
使用RNN进行多分类，识别a第一次出现在字符串中的位置

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)    #embedding层
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=128, batch_first=True)
        self.classify = nn.Linear(128, sentence_length)  # 输出维度=序列长度
        self.loss = nn.CrossEntropyLoss()   #loss函数用交叉熵，自动先过一遍softmax

    def forward(self, x, y=None):
        # Embedding层
        x = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, vector_dim)

        # RNN处理
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size)

        # 位置预测（每个时间步输出对应位置分数）
        y_pred = self.classify(rnn_out)  # (batch, seq_len, sentence_length)
        y_pred = y_pred.mean(dim=1)  # 平均各时间步特征

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.argmax(y_pred, dim=1)


def build_vocab():
    # 确保包含字母a和其他必要字符
    chars = "abcdefghijklmnopqrstuvwxyz你我他"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 生成必须包含a的字符串
    chars = list(vocab.keys())
    chars.remove('pad')  # 排除填充符
    chars.remove('unk')  # 排除未知符

    # 保证至少包含一个a
    x = []
    has_a = False
    for _ in range(sentence_length):
        char = random.choice(chars)
        x.append(char)
        if char == 'a':
            has_a = True

    # 强制插入a的逻辑
    if not has_a:
        idx = random.randint(0, sentence_length - 1)
        x[idx] = 'a'

    # 查找第一个a的位置
    first_a_pos = next(i for i, c in enumerate(x) if c == 'a')

    # 转换为索引序列
    x = [vocab.get(c, vocab['unk']) for c in x]
    return x, first_a_pos


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 直接存储整数标签
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    print("真实标签示例：", y[:10].tolist())

    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        correct = (y_pred == y).sum().item()

    accuracy = correct / len(y)
    print("正确预测个数：%d，准确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 50  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.001  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


if __name__ == "__main__":
    main()
