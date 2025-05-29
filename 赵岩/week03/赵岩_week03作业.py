#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中'a'第一次出现的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)          # RNN层
        self.classify = nn.Linear(vector_dim, sentence_length)              # 分类层，输出每个位置的可能性
        self.activation = torch.softmax                                       # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()                                   # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        out, _ = self.rnn(x)                       # (batch_size, sen_len, vector_dim)
        x = out[:, -1, :]                          # 取最后一个时间步的输出 (batch_size, vector_dim)
        y_pred = self.classify(x)                  # (batch_size, sentence_length)
        y_pred = self.activation(y_pred, dim=1)    # 归一化成概率分布
        if y is not None:
            return self.loss(y_pred, y)            # 预测值和真实值计算损失
        else:
            return y_pred.argmax(dim=1)            # 返回预测结果


# 字符集只用英文字母，便于测试
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本：插入一个 'a'，返回字符串的index表示以及第一个'a'的位置
def build_sample(vocab, sentence_length):
    # 随机选 sentence_length-1 个字符
    x_chars = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
    insert_pos = random.randint(0, len(x_chars))
    x_chars.insert(insert_pos, 'a')  # 插入一个'a'
    x_ids = [vocab.get(word, vocab['unk']) for word in x_chars]
    first_a_pos = x_chars.index('a')
    return x_ids[:sentence_length], first_a_pos


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码：测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


# 主函数：训练模型
def main():
    # 配置参数
    epoch_num = 15         # 训练轮数
    batch_size = 32        # 每次训练样本个数
    train_sample = 1000    # 每轮训练总共训练的样本总数
    char_dim = 20          # 每个字的维度
    sentence_length = 6    # 样本文本长度
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()

    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, vocab, sentence_length)

    # 保存模型
    torch.save(model.state_dict(), "rnn_position_model.pth")

    # 保存词表
    writer = open("rnn_position_vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    sentence_length = 6
    char_dim = 20
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)       # 初始化模型结构
    model.load_state_dict(torch.load(model_path))               # 加载模型权重
    model.eval()

    for input_string in input_strings:
        # 将字符串转换为 token id
        x = [vocab.get(c, vocab['unk']) for c in input_string]
        x = x[:sentence_length]
        x = x + [0] * (sentence_length - len(x))  # padding
        with torch.no_grad():
            result = model(torch.LongTensor([x]))[0]
            print("输入：%s, 预测第一个'a'的位置：%d" % (input_string, int(result)))


if __name__ == "__main__":
    main()
    test_strings = ["abcxyz", "bcdefa", "abcdef", "xyzabc"]
    predict("rnn_position_model.pth", "rnn_position_vocab.json", test_strings)
