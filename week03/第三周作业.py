# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from torch.nn.utils.rnn import pad_sequence

"""
基于pytorch的RNN网络编写
实现一个多分类任务：判断字符'a'第一次出现在字符串中的位置
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, num_classes)  # 线性层
        self.activation = torch.softmax  # 使用softmax进行多分类
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失
        self.sentence_length = sentence_length

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        _, h_n = self.rnn(x)  # RNN处理，获取最后一个时间步的隐藏状态
        x = h_n.squeeze(0)  # 去除多余的维度
        x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, num_classes)
        y_pred = self.activation(x, dim=1)  # 使用softmax得到概率分布

        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 字符集
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机生成字符串，确保至少包含一个'a'
    while True:
        x = [random.choice(list(vocab.keys())[1:-1]) for _ in range(sentence_length)]
        if 'a' in x:
            break

    # 找到第一个'a'的位置（从0开始）
    first_a_pos = x.index('a')
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号

    return x, first_a_pos


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    # 对序列进行填充以保证长度一致
    dataset_x = pad_sequence([torch.LongTensor(x) for x in dataset_x], batch_first=True, padding_value=0)
    return dataset_x, torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchModel(char_dim, sentence_length, vocab, num_classes)
    return model


# 测试代码
def evaluate(model, vocab, sample_length, num_classes):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本

    # 统计每个类别的样本数
    class_counts = [0] * num_classes
    for label in y:
        class_counts[label.item()] += 1
    print("各类别样本数:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred.data, 1)  # 获取预测类别
        correct = (predicted == y.squeeze()).sum().item()
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 32  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    num_classes = sentence_length  # 类别数为字符串长度（a可能出现在任何位置）
    learning_rate = 0.001  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)
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
        acc = evaluate(model, vocab, sentence_length, num_classes)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 32  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    num_classes = sentence_length
    model = build_model(vocab, char_dim, sentence_length, num_classes)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    # 预处理输入数据
    x = []
    for input_string in input_strings:
        # 只保留前sentence_length个字符
        truncated = input_string[:sentence_length]
        # 转换为ID序列
        ids = [vocab.get(char, vocab['unk']) for char in truncated]
        # 填充到固定长度
        padded = ids + [0] * (sentence_length - len(ids))
        x.append(padded)

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测

    for i, input_string in enumerate(input_strings):
        probs = result[i]
        predicted_class = torch.argmax(probs).item()
        print(f"输入：{input_string}")
        print(f"预测'a'首次出现位置：{predicted_class}")
        print(f"位置概率分布：{probs.tolist()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
    test_strings = ["banana", "apple", "cards", "aaaaaa", "cbadfg", "alone"]
    predict("rnn_model.pth", "vocab.json", test_strings)
