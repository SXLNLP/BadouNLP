# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的RNN网络实现多分类任务
判断字符'a'在字符串中出现的位置（类别）
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 损失函数采用交叉熵损失
        self.sentence_length = sentence_length

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim)
        x = self.classify(rnn_out)  # (batch_size, sen_len, num_classes)

        if y is not None:
            # 将y从(batch_size, sen_len)变为(batch_size * sen_len)
            # 将x从(batch_size, sen_len, num_classes)变为(batch_size * sen_len, num_classes)
            x = x.view(-1, x.size(-1))  # (batch_size * sen_len, num_classes)
            y = y.view(-1)  # (batch_size * sen_len)

            # 计算损失时忽略padding位置
            mask = (y != -100)  # 假设我们用-100表示padding位置
            x = x[mask]
            y = y[mask]

            return self.loss(x, y)
        else:
            # 预测时返回每个位置的softmax概率
            return torch.softmax(x, dim=2)


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = [0] * sentence_length  # 初始化所有位置为0

    # 随机选择一个位置放入'a'，或者不放
    if random.random() < 0.5:
        a_pos = random.randint(0, sentence_length - 1)
        x[a_pos] = 'a'
        y[a_pos] = 1  # 标记'a'的位置为1

    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    num_classes = 2  # 二分类：每个位置是否包含'a'
    model = TorchModel(char_dim, sentence_length, vocab, num_classes)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)

    # 统计正负样本数量
    pos_samples = (y == 1).sum().item()
    neg_samples = (y == 0).sum().item()
    print(f"本次预测集中共有{pos_samples}个正样本，{neg_samples}个负样本")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # (batch_size, sen_len, 2)
        y_pred = y_pred.argmax(dim=2)  # 获取预测类别

        # 计算准确率
        correct = (y_pred == y).sum().item()
        total = y.numel()
        accuracy = correct / total

    print(f"正确预测个数：{correct}, 总预测数：{total}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
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
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    # 处理输入字符串
    x = []
    for s in input_strings:
        # 截断或填充到固定长度
        s = s[:sentence_length].ljust(sentence_length, ' ')
        x.append([vocab.get(c, vocab['unk']) for c in s])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        preds = result.argmax(dim=2)  # 获取预测类别

    for i, s in enumerate(input_strings):
        print(f"\n输入: {s}")
        print("预测结果:")
        for pos, char in enumerate(s[:sentence_length]):
            if preds[i][pos].item() == 1:
                print(f"  '{char}'在位置{pos}被预测为'a'")


if __name__ == "__main__":
    main()
    test_strings = ["abcdef", "xyzaxy", "nolett", "aaasaa"]
    predict("rnn_model.pth", "vocab.json", test_strings)
