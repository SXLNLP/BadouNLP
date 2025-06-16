# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现RNN网络完成多分类任务
判断字符'a'在字符串中第一次出现的位置

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, 128, batch_first=True)  # RNN层
        self.classify = nn.Linear(128, num_classes)  # 线性层输出多分类结果
        self.loss = nn.CrossEntropyLoss()  # 多分类使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        _, h_n = self.rnn(x)  # 使用RNN处理序列
        h_n = h_n.squeeze(0)  # 去除RNN的num_layers维度
        y_pred = self.classify(h_n)  # (batch_size, num_classes)

        if y is not None:
            y = y.squeeze().long()  # 确保标签是长整型
            return self.loss(y_pred, y)  # 计算交叉熵损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 输出概率分布


# 字符集包含所有需要的字符
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 英文字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机生成字符串（确保包含至少一个'a'）
    if random.random() < 0.3:  # 30%概率生成不含'a'的样本
        chars = [random.choice(list(vocab.keys())[1:-1]) for _ in range(sentence_length)]
        while 'a' in chars:  # 确保不含'a'
            chars = [random.choice(list(vocab.keys())[1:-1]) for _ in range(sentence_length)]
        label = sentence_length  # 未出现标记为最后一位
    else:
        # 随机确定'a'首次出现的位置
        first_a_pos = random.randint(0, sentence_length - 1)
        chars = []
        for i in range(sentence_length):
            if i == first_a_pos:
                chars.append('a')
            else:
                # 避免在'a'之前出现'a'
                char = random.choice(list(vocab.keys())[1:-1])
                while char == 'a':
                    char = random.choice(list(vocab.keys())[1:-1])
                chars.append(char)
        label = first_a_pos

    # 将字符转换为索引
    x = [vocab.get(char, vocab['unk']) for char in chars]
    return x, label


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchModel(char_dim, sentence_length, vocab, num_classes)
    return model


# 测试模型准确率
def evaluate(model, vocab, sample_length, num_classes):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)

    # 统计各类别样本数
    class_count = [0] * (num_classes + 1)
    for label in y:
        class_count[label.item()] += 1

    print(f"各类别样本数: {class_count[0:num_classes]} (未出现:{class_count[num_classes]})")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 获取预测概率分布
        predicted_labels = torch.argmax(y_pred, dim=1)  # 获取预测类别

        for pred, true in zip(predicted_labels, y):
            if pred.item() == true.item():
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print(f"正确预测: {correct}, 错误预测: {wrong}, 正确率: {accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 批次大小
    train_sample = 3000  # 每轮训练样本数
    char_dim = 64  # 字符嵌入维度
    sentence_length = 10  # 字符串长度
    num_classes = sentence_length + 1  # 类别数（位置0-9 + 未出现）
    learning_rate = 0.001  # 学习率

    # 建立字表
    vocab = build_vocab()
    print("字符表大小:", len(vocab))

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    log = []
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

        avg_loss = np.mean(watch_loss)
        print(f"Epoch {epoch + 1}/{epoch_num}, 平均loss: {avg_loss:.6f}")
        acc = evaluate(model, vocab, sentence_length, num_classes)
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")

    # 保存词表
    with open("rnn_vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 测试示例
    test_strings = [
        "bcdefghijk",  # 无'a'
        "a123456789",  # 位置0
        "0a23456789",  # 位置1
        "01a3456789",  # 位置2
        "012a456789",  # 位置3
        "0123a56789",  # 位置4
        "01234a6789",  # 位置5
        "012345a789",  # 位置6
        "0123456a89",  # 位置7
        "01234567a9",  # 位置8
        "012345678a",  # 位置9
    ]
    predict("rnn_model.pth", "rnn_vocab.json", test_strings, sentence_length, num_classes)


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings, sentence_length, num_classes):
    char_dim = 64
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    model.load_state_dict(torch.load(model_path))

    # 处理输入
    x = []
    for s in input_strings:
        # 截断或填充到固定长度
        s = s[:sentence_length].ljust(sentence_length, 'z')
        x.append([vocab.get(char, vocab['unk']) for char in s])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        predicted_classes = torch.argmax(result, dim=1)

    print("\n预测结果:")
    for i, s in enumerate(input_strings):
        pred = predicted_classes[i].item()
        if pred == sentence_length:
            print(f"字符串: '{s}' -> 'a'未出现")
        else:
            print(f"字符串: '{s}' -> 'a'首次出现在位置 {pred}")


if __name__ == "__main__":
    main()
