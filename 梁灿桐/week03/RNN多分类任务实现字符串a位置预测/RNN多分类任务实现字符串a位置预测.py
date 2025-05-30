# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from torch.utils.data import TensorDataset, DataLoader

"""

基于pytorch的RNN网络实现多分类任务
判断字符串中字母'a'第一次出现的位置，未出现则输出“未出现”

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab,hidden_size=128,num_layers=2):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'
        )
        self.classify = nn.Linear(hidden_size, sentence_length + 1)   # 分类层：输出大小为 sentence_length + 1（0到sentence_length共sentence_length+1类）
        self.loss = nn.CrossEntropyLoss()   # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)           # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)         # (batch_size, sen_len, hidden_size)
        output = output[:, -1, :]       # 取最后一个时间步的输出 (batch_size, hidden_size)
        y_pred = self.classify(output)  # (batch_size, num_classes)
        if y is not None:
            y = y.squeeze()             # (batch_size, 1) -> (batch_size,)
            return self.loss(y_pred, y.long())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 字符集：小写字母 + 特殊字符
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机生成字符串（可能包含或不包含'a'）
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 找出第一个'a'的位置
    target = sentence_length  # 默认值为字符串长度（表示未出现）
    for i, char in enumerate(x):
        if char == 'a':
            target = i
            break
    # 将字符转换为索引
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, target


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    y = y.squeeze()
    # 统计各类别样本数
    class_counts = [0] * (sample_length + 1)
    for target in y:
        class_counts[target] += 1
    print("类别分布:", ", ".join([f"{i}:{count}" for i, count in enumerate(class_counts)]))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=20)
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"正确预测个数: {correct}, 总数: {total}, 正确率: {correct / total:.4f}")
    return correct / total


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.001  # 学习率
    # 建立字表
    vocab = build_vocab()
    print("字符表:", vocab)
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 构建训练数据集
        train_x, train_y = build_dataset(train_sample, vocab, sentence_length)
        dataset = TensorDataset(train_x, train_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for inputs, targets in loader:
            optim.zero_grad()  # 梯度归零
            loss = model(inputs, targets) # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        # print(f"Epoch {epoch + 1}/{epoch_num}, 平均loss: {np.mean(watch_loss):.6f}")

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
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # 将输入字符串转换为索引序列
    x = []
    for s in input_strings:
        seq = [vocab.get(c, vocab['unk']) for c in s]
        # 填充或截断到固定长度
        if len(seq) < sentence_length:
            seq += [0] * (sentence_length - len(seq))
        else:
            seq = seq[:sentence_length]
        x.append(seq)

    model.eval()
    with torch.no_grad():
        x = torch.LongTensor(x)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    for i, s in enumerate(input_strings):
        pred_class = predictions[i].item()
        prob = probs[i, pred_class].item()
        # 根据预测类别显示不同信息
        if pred_class == sentence_length:
            print(f"输入: '{s}'\t预测结果: 未出现")
        else:
            print(f"输入: '{s}'\t预测结果: 位置{pred_class + 1}")


if __name__ == "__main__":
    main()
    # test_strings = [
    #     "bcdefghijk",  # 无a -> 预测10 -> "未出现"
    #     "xyzabcmno",   # a在位置4
    #     "noahere",     # a在位置3
    #     "aaaaa",       # a在位置1
    #     "nothing"      # 无a -> 预测10 -> "未出现"
    # ]
    # predict("model.pth", "vocab.json", test_strings)
