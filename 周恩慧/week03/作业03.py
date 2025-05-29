# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128, num_layers=2):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        # 使用LSTM作为RNN层
        self.rnn = nn.LSTM(vector_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
        # 分类层，输出大小为sentence_length+1（0表示未出现）
        self.classify = nn.Linear(hidden_size, sentence_length + 1)
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # 通过RNN层
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, hidden_size)
        # 取最后一个时间步的输出作为整个序列的表示
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        # 分类
        y_pred = self.classify(last_output)  # (batch_size, sentence_length+1)

        if y is not None:
            # 将y转换为长整型，因为CrossEntropyLoss要求这样的输入
            y = y.squeeze().long()
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 只使用小写字母
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 随机生成字符串
    chars = list(vocab.keys())
    chars.remove('pad')
    chars.remove('unk')

    x = [random.choice(chars) for _ in range(sentence_length)]
    position = 0  # 默认未出现

    # 检查'a'第一次出现的位置
    for i, char in enumerate(x):
        if char == 'a':
            position = i + 1  # 位置从1开始计数
            break

    # 将字符转换为索引
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, position


def build_dataset(sample_length, vocab, sentence_length):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append([y])
    return torch.LongTensor(X), torch.LongTensor(Y)


def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    y = y.squeeze()  # 去掉多余的维度


    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 64
    train_sample = 3000
    char_dim = 50
    sentence_length = 10  # 字符串长度
    learning_rate = 0.001

    # 建立字表
    vocab = build_vocab()
    print("字符表:", vocab)

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程
    log  = []

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




def predict(model_path, vocab_path, input_strings):
    char_dim = 50
    sentence_length = 10

    # 加载字符表
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表

    # 反转vocab，用于输出
    idx_to_char = {idx: char for char, idx in vocab.items()}

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 处理输入
    x = []
    for s in input_strings:
        # 截断或填充到固定长度
        s = s.lower()[:sentence_length].ljust(sentence_length, 'z')  # 用'z'填充不足部分
        # 转换为索引
        encoded = [vocab.get(char, vocab['unk']) for char in s]
        x.append(encoded)

    # 预测
    with torch.no_grad():
        tensor_x = torch.LongTensor(x)
        y_pred = model(tensor_x)
        predictions = torch.argmax(y_pred, dim=1)

    # 打印结果
    for i, s in enumerate(input_strings):
        pred_pos = predictions[i].item()
        print("输入：%s, 预测位置：%d" % (s, pred_pos))  # 打印结果


if __name__ == "__main__":
    # 训练模型
    log = main()

    # 测试预测
    test_strings = [
        "abcdefghij",  # a在位置1
        "bbacdefghi",  # a在位置3
        "xyzaxyzaxy",  # a在位置4
        "aabbccddee",  # a在位置1
        "zzzzzzzzza",  # a在位置10
        "1hello3aworld"  # 包含非字母字符,a在位置8
    ]

    predict("model.pth", "vocab.json", test_strings)
