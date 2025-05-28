# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
输出文本中类别为a第一次出现在字符串中的位置

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, out_size, sen_len, vocab):
        super(TorchModel, self).__init__()
        self.out_size = out_size
        self.sen_len = sen_len
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.rnn = nn.RNN(vector_dim, hidden_size=128, batch_first=True, num_layers=2, dropout=0.4) # 输入的向量维度为vector_dim 隐藏层的维度为hidden_size batch_first=True表示张量形状为(batch_size, sen_len, vector_dim)
        self.classify = nn.Linear(128, out_size)  # 线性层
        # self.activation = torch.sigmoid  # sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size), (1, sen_len, hidden_size)
        x = x[:, -2, :]  # *** (batch_size, sen_len, hidden_size) -> (batch_size, hidden_size) -> batch_size个(sen_len, hidden_size)的张量 全部取倒数第2个字符然后再组成(batch_size, hidden_size)的张量
        y_pred = self.classify(x)  # (batch_size, hidden_size) -> (batch_size, out_size)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他abcfdsafdsdefghijklmnopqrstuvwxyz"  # 字符集
    chars_set = set(chars)
    vocab = {"pad": 0}
    for index, chars_set in enumerate(chars_set):
        vocab[chars_set] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 放在最后
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x[random.randint(0, sentence_length - 1)] = "a"
    first_index = x.index("a")
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, first_index


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, out_size, sentence_length):
    model = TorchModel(char_dim, out_size, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        pred_labels = y_pred.argmax(dim=1)  # (batch_size,)
        correct = (pred_labels == y).sum().item()
        total = y.size(0)
        accuracy = correct / total
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 15  # 每个字的维度
    out_size = 6  # 输出的向量维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率
    # # 建立字表
    vocab = build_vocab()
    # vocab = json.load(open('vocab.json', 'r', encoding='utf-8'))
    # 建立模型
    model = build_model(vocab, char_dim, out_size, sentence_length)
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
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 15  # 每个字的维度
    out_size = 6  # 输出向量的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, out_size, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s, 概率和为：%f" % (input_string, result[i].argmax().item(), result[i], result[i].sum()))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fnafea", "az你aaa", "rqaagj", "n我kwaa"]
    predict("model.pth", "vocab.json", test_strings)
