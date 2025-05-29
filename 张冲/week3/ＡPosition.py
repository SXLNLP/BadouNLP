# 构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。


import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # self.pool = nn.MaxPool1d(sentence_length)  # 池化层
        self.classify = nn.RNN(input_size=vector_dim, hidden_size=hidden_size, batch_first=True)  # RNN层
        self.liner = nn.Linear(hidden_size, sentence_length + 1)
        self.activation = torch.softmax  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x, h = self.classify(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        y_pred = self.liner(x[:, -1, :])  # (batch_size, sen_len, hidden_size) -> (batch_size, sen_len + 1)
        y_pred = self.activation(y_pred, dim=-1)  # (batch_size, sen_len + 1) -> (batch_size, sen_len + 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqr"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


def find_index(list, val):
    for i, v in enumerate(list):
        if v == val:
            return i
    return -1


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = np.zeros(sentence_length + 1)
    a_index = find_index(x, 'a')
    y[a_index + 1] = 1
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1
            # if abs(y_p - y_t) < 0.5:
            #     correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


char_dim = 20  # 每个字的维度
sentence_length = 10
# 样本文本长度
hidden_size = 128


def train(vocab_path, model_path):
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)
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
    torch.save(model.state_dict(), model_path)
    # 保存词表
    writer = open(vocab_path, "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    print(vocab)
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x, y_true = build_dataset(100, vocab, sentence_length)
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        y_pred = model.forward(torch.LongTensor(x))  # 模型预测
    for x_item, y_t, y_p in zip(x, y_true, y_pred):
        vec_max_index = np.argmax(y_t)
        res_max_index = np.argmax(y_p)
        print("输入：%s, 实际分类%d,  预测概率最大分类%d,  概率为%f" % (
            x_item, vec_max_index, res_max_index, y_p[res_max_index]))  # 打印结果


if __name__ == "__main__":
    vocab_path = 'vocab.train'
    model_path = 'model.train'
    # train(vocab_path, model_path)
    predict(model_path, vocab_path)
