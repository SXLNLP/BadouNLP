import torch
import torch.nn as nn
import numpy as np
import random
import json

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, num_classes)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类
        '''
        只有20正确率
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.classify = nn.Linear(vector_dim, num_classes)  # 线性层，输出维度为类别数
        self.activation = torch.softmax  # softmax归一化函数，将输出转换为概率分布
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失
        '''

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)  # RNN处理，输入形状(batch_size, sen_len, input_dim)output形状(batch_size, sen_len, vector_dim)
        # 取最后一个时间步的输出作为分类依据
        x = output[:, -1, :]  # (batch_size, sequence_length, hidden_size) 变为 (batch_size, hidden_size)
        y_pred = self.classify(x)  # (batch_size, num_classes)

        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出概率分布

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 随机生成字符串
    x = [random.choice(list(vocab.keys())[1:-1]) for _ in range(sentence_length)]

    # 确定'a'第一次出现的位置（类别）
    if 'a' in x:
        y = x.index('a')  # 位置索引作为类别
    else:
        y = sentence_length  # 如果没有'a'，类别为字符串长度

    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchModel(char_dim, sentence_length, vocab, num_classes)
    return model


def evaluate(model, vocab, sample_length, num_classes):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)

    # 统计各类别样本数量
    class_counts = [0] * (num_classes + 1)
    for label in y:
        class_counts[label.item()] += 1
    print("各类别样本数量:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        predicted_classes = torch.argmax(y_pred, dim=1)
        correct = (predicted_classes == y.squeeze()).sum().item()
        wrong = len(y) - correct

    print(f"正确预测个数: {correct}, 正确率: {correct / (correct + wrong):.4f}")
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 3200  # 每轮训练总共训练的样本总数
    char_dim = 32  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.001  # 学习率
    num_classes = sentence_length + 1  # 类别数（a位置0到sentence_length-1,有len种类别，总类别+1种没出现的情况）

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
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, sentence_length, num_classes)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return log


def predict(model_path, vocab_path, input_strings):
    char_dim = 32
    sentence_length = 10
    num_classes = sentence_length + 1

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    model.load_state_dict(torch.load(model_path))

    # 处理输入字符串
    x = []
    for s in input_strings:
        # 截断或填充到固定长度
        s = s[:sentence_length].ljust(sentence_length, 'z')
        x.append([vocab.get(char, vocab['unk']) for char in s])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        predicted_classes = torch.argmax(result, dim=1)

    for i, s in enumerate(input_strings):
        pred_pos = predicted_classes[i].item()
        if pred_pos == sentence_length:
            print(f"输入: '{s}', 预测: 'a'未出现")
        else:
            print(f"输入: '{s}', 预测: 'a'首次出现在位置 {pred_pos}")


if __name__ == "__main__":
    main()
    # test_strings = ["bcdefghiaj", "bcdaefghij", "bbccddeeff", "bbccaaddee", "qwesddffrt"]
    # predict("rnn_model.pth", "vocab.json", test_strings)
