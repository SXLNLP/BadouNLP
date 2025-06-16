# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json


# 定义七分类模型
class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, 7)  # 线性层，7个类别
        self.loss = nn.functional.cross_entropy  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim)
        x = rnn_out[:, -1, :]  # (batch_size, vector_dim)
        x = self.classify(x)  # (batch_size, 7)
        if y is not None:
            return self.loss(x, y.view(-1).long())  # 计算损失
        else:
            return x  # 返回预测结果


# 建立词汇表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789你我他"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 处理未知字符
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    a_positions = [i for i, char in enumerate(x) if char == 'a']  # 找到'a'的位置
    if len(a_positions) == 0:
        y = 6  # 没有'a'，类别为6（从0开始索引）
    else:
        y = a_positions[0]  # 第一次出现位置

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号
    return x, y


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
def build_model(vocab, char_dim, sentence_length):
    model = RNNModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 获取预测类别
        for y_t, y_p in zip(y.view(-1), predicted):
            if y_t == y_p:
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
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
    torch.save(model.state_dict(), "model3.pth")
    # 保存词表
    with open("vocab3.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重

    x = []
    for input_string in input_strings:
        if len(input_string) != sentence_length:
            print(f"输入字符串 '{input_string}' 长度不正确，应为 {sentence_length}，请调整。")
            continue
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化

    if not x:  # 如果没有有效的输入字符串
        print("没有有效的输入字符串进行预测。")
        return

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        _, predicted = torch.max(result, 1)  # 获取预测类别

    for i, input_string in enumerate(input_strings):
        if len(input_string) == sentence_length:  # 只打印正确长度的输入
            print("输入：%s, 预测类别：%d" % (input_string, predicted[i].item()))  # 打印结果


if __name__ == "__main__":
    # main()
    test_strings = ["abcdef", "bcdefg", "aaaaaa", "defabc", "fghija", "abadfg", "efghha"]
    predict("model3.pth", "vocab3.json", test_strings)
