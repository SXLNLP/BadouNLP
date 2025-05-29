"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法
"""

# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的RNN网络编写
实现RNN完成多分类任务.字符串为定长6
预测字符'a'第一次出现在字符串中的位置（0-5）
"""


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(RNNModel, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # RNN层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        # 线性层，输出6个类别
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        # 交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        # RNN处理序列
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]
        # 分类
        y_pred = self.classify(x)
        if y is not None:
            # 计算损失
            return self.loss(y_pred, y)
        else:
            # 输出预测结果
            return nn.Softmax(dim=1)(y_pred)


def build_vocab():
    # 字符集
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {'pad': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    # 指定哪些字出现时为正样本
    if 'a' in x:
        y = x.index('a')
    else:
        y = sentence_length
        # 将字转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        # 标签是位置索引（0-5）
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    return RNNModel(char_dim, sentence_length, vocab)


def evaluate(model, vocab, sample_length):
    model.eval()
    # 建立200个用于测试的样本
    x, y = build_dataset(200, vocab, sample_length)
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        # 模型预测
        y_pred = model(x)
        # 对比下标索引
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 40  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 30  # 每个字的维度
    sentence_length = 6  # 样本文本长度
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
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss).item()))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()


def predict(model_path, vocab_path, input_strings):
    char_dim = 30
    sentence_length = 6
    # 加载词汇表
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    # 测试模式
    model.eval()
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化 此处支持正常数据，出现异常数据（!=6）会出错，可以采用截断
    with torch.no_grad():
        res = model.forward(torch.LongTensor(x))  # 模型预测
        for i, input_string in enumerate(input_strings):
            # 打印结果
            print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(res[i]).item(), res[i]))


if __name__ == "__main__":
    # 训练模型
    # main()
    # 测试字符串
    test_strings = ["fnabex", "dfwahg", "rjclag", "akplmn", "xyzsda", "bachqt"]
    predict("model.pth", "vocab.json", test_strings)
