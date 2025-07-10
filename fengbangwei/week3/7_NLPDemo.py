# -*- coding: utf-8 -*-
"""
@Time ： 2025/5/27 15:07
@Auth ： fengbangwei
@File ：7_NLPDemo.py

"""

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的网络编写
构造随机包含a的字符串，使用rnn进行多分类，
类别为a第一次出现在字符串中的位置。
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_dim):
        super(TorchModel, self).__init__()
        # 初始化  29个字符 词向量维度是20
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # Embedding层
        self.rnn = nn.RNN(vector_dim, hidden_dim, bias=False, batch_first=True)  # RNN层 20 x 40
        self.classify = nn.Linear(hidden_dim, sentence_length)  # 线性层 40 x 6
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉损失 进行多分类

    def forward(self, x, y=None):
        # print(x.shape)  # torch.Size([20, 6])
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # print(x.shape)  # torch.Size([20, 6, 20])
        out, hidden = self.rnn(x)  # (batch_size, vector_dim) -> (batch_size, 1) 20*20 20*1 -> 20*1
        # print(out.shape)  # torch.Size([20, 6, 40])
        # print(hidden.shape)  # torch.Size([1, 20, 40])
        # 取最后一个时间步的输出做分类
        final_time_step = out[:, -1, :]
        # print(final_time_step.shape)  # torch.Size([20, 40])
        y_pred = self.classify(final_time_step)
        # print(y_pred.shape)  # torch.Size([20, 6])
        # 当输入真实标签，返回loss值；无真实标签，返回预测值
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


#  建立字表
def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机样本
def build_sample(vocab, sentence_length):
    chars = "a你a我a他bacadeafagahaijkalamanaopqarasatauavawaxyaz"  # 字符集
    target_chars = [random.choice(chars) for _ in range(sentence_length)]
    if target_chars.__contains__('a'):
        index = target_chars.index('a')
        y = 5 if index == 5 else index # 如果a的位置是5，则预测结果为5 表示预留最有一个位置给其他类别
    else:
        y = 5  # 不包含a，则预测结果为5 表示其他类别
    x = [vocab.get(char, vocab['unk']) for char in target_chars]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    '''
    :param sample_length:  20
    :param vocab:
    :param sentence_length: 6
    :return:
    '''
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 构建模型
def build_model(vocab, char_dim, sentence_length, hidden_dim):
    model = TorchModel(char_dim, sentence_length, vocab, hidden_dim)
    return model


# 测试模型
def evaluate(model, vocab, sentence_length):
    # 测试模式
    model.eval()
    # 构建200个数据 预测看看
    x, y = build_dataset(200, vocab, sentence_length)
    wrong_cnt = (y == 5).sum().item()
    print("本次预测集中共有%d个正样本，%d个负样本" % (200 - wrong_cnt, wrong_cnt))
    with torch.no_grad():
        # 预测
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)  # 获取预测类别
        correct = (predicted == y).sum().item()  # 直接比较所有预测与真实标签
        wrong = len(y) - correct

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / len(y)))
    return correct / (correct + wrong)


#  预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_dim = 40  # 隐藏层维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length, hidden_dim)  # 建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重  防止潜在的反序列化攻击
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 对输入的字符串进行编码
    model.eval()  # 测试模式
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))  # 模型预测

    # print(result)
    # print(result.shape)
    for i, input_string in enumerate(input_strings):
        # 假设 result 是二维张量 [batch_size, num_classes]
        prob, predicted = torch.max(result[i].unsqueeze(0), 1)  # 添加 batch 维度
        print("输入：%s, 预测类别：%d, 概率值：%f, 全部概率值：%s" % (
            input_string, predicted.item(), prob.item(), result[i].tolist()))


def mian():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    hidden_dim = 40  # 隐藏层维度
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    # 10 轮 每轮训练 25次 总共 250次
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        #  500 / 20 = 25 每轮 训练 25 次
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 生成20个样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 测试本轮模型结果
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("rnn_vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


if __name__ == '__main__':
    mian()
    # test_strings = ["fav他ae", "wz你dfg", "rqadeg", "ajn我kw","他qwejk"]
    # predict("rnn_model.pth", "rnn_vocab.json", test_strings)
    # x,y = build_dataset(200, build_vocab(), 6)
    # print(x)
    # print(y)
