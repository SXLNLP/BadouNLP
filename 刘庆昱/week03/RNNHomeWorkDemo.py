# coding:utf8
import math
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
输入一个字符串，根据字符a所在位置进行分类
使用RNN对字符串进行多分类 
类别为 'a' 第一次出现在字符串中的位置

发现的问题：
对字符串 "aaaaaaaaaa" 的分类不稳定
准确的说是对有多个 'a' 的字符串的判定都不稳定

问题的解决：
build_sample 样本生成器建立的时候要排除 “pad” 和 “unk” 来建立合理的训练集来进行正确的训练

"""


# 建立词表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 'a' 对应值为1
    vocab["unk"] = len(vocab)  # 最终值为27（26字母+unk）
    return vocab


# 建立样本生成器
def build_sample(vocab, sentence_length):
    # 1. 将词表转换为列表
    li_chars = list(vocab.keys())
    # 2. 移除掉pad和unk，方便生成样本
    li_chars.remove("pad")
    li_chars.remove("unk")
    # 3. 随机生成一个样本列表
    sentence = [random.choice(li_chars) for _ in range(sentence_length)]
    # 4. 选取第一个a出现的位置为y
    if "a" in sentence:
        y = sentence.index("a")
    else:
        y = sentence_length
    # 5. 将样本列表转换为索引列表
    x = [vocab.get(char, vocab["unk"]) for char in sentence]
    return x, y


def build_dataset(dataset_len, vocab, sentence_len):
    dataset_x = []
    dataset_y = []
    for i in range(dataset_len):
        x, y = build_sample(vocab, sentence_len)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, dropout_rate, rnn_hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # Embedding层
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层
        self.rnn = nn.RNN(  # RNN层
            input_size=vector_dim,
            hidden_size=rnn_hidden_size,
            batch_first=True
        )
        self.linear = nn.Sequential(  # 线性层
            nn.Linear(rnn_hidden_size, 1024),  # (batch_size, hidden_size) -> (batch_size, 1024)
            nn.LayerNorm(1024),  # (batch_size, 1024) -> (batch_size, 1024)
            nn.GELU(),  # (batch_size, 1024) -> (batch_size, 1024)
            nn.Dropout(dropout_rate),  # (batch_size, 1024) -> (batch_size, 1024)
            nn.Linear(1024, 512),  # (batch_size, 1024) -> (batch_size, 512)
            nn.LayerNorm(512),  # (batch_size, 512) -> (batch_size, 512)
            nn.GELU(),  # (batch_size, 512) -> (batch_size, 512)
            nn.Linear(512, sentence_length + 1)  # (batch_size, 512) -> (batch_size, sentence_length)
        )
        self.activation = nn.ReLU
        self.loss = nn.functional.cross_entropy  # loss函数采用 交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.dropout(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        rnn_out, hidden = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        x = rnn_out[:, -1, :]  # (batch_size, sen_len, hidden_size) -> (batch_size, hidden_size)
        # x = self.linear1(x)
        y_pred = self.linear(x)  # (batch_size, hidden_size) -> (batch_size, sentence_len)
        if y is not None:
            return self.loss(y_pred, y)  # 训练模式 预测值和真实值计算损失
        else:
            return y_pred  # 预测模式 输出预测结果


# 建立模型
def build_model(vector_dim, sentence_length, vocab, dropout_rate, rnn_hidden_size):
    model = RNNModel(vector_dim, sentence_length, vocab, dropout_rate, rnn_hidden_size)
    return model


# 测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print("标签分布:", Counter(y.numpy()))  # 查看类别分布
    correct = 0
    with torch.no_grad():
        logits = model(x)  # 输出logits形状: (200, sample_length)
        preds = logits.argmax(dim=1)  # 取预测类别 (200,)

        correct = (preds == y).sum().item()
        acc = correct / len(y)

    print(f"正确预测数: {correct}, 正确率: {acc:.4f}")
    return acc


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 40  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    vector_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.001  # 学习率
    dropout_rate = 0.1
    rnn_hidden_size = 1024  # RNN隐藏层维度
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vector_dim, sentence_length, vocab, dropout_rate, rnn_hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(math.ceil(train_sample / batch_size)):
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


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    vector_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    dropout_rate = 0.1
    rnn_hidden_size = 1024
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vector_dim, sentence_length, vocab, dropout_rate, rnn_hidden_size)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        # 1. 字符截断/填充处理
        chars = list(input_string)[:sentence_length]  # 截断超长部分
        chars += ['pad'] * (sentence_length - len(chars))  # 填充不足部分
        # 2. 转换索引 (处理未知字符)
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fnvfeaa   ", "wadf sfagg", "rawdqwfdeg", "nlwsgewgas", "sfafr3qwer", "aaaaaaaaaa", "ababababab",
                    "kijabcdefh", "gijkbcdeaf", "gkijadabec", "kijhdefacb", "bsfwggdbdg"]
    predict("model.pth", "vocab.json", test_strings)
