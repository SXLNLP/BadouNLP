# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json

"""  基于pytorch的RNN实现文本多分类(a出现的位置就是它的类别)  """

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=64):
        super(TorchModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # LSTM层
        self.lstm = nn.LSTM(input_size=vector_dim, hidden_size=hidden_size, batch_first=True)
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=hidden_size, batch_first=True)
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, sentence_length)
        self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        # x, _ = self.rnn(x)
        x, _ = self.lstm(x)

        y_pred = self.fc(x[:, -1, :])
        if y is not None:
            return self.loss(y_pred, y.reshape(-1))  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length-1)]
    x.append('a')
    random.shuffle(x)
    try:
        y = x.index('a')   # 设置样本的类别
        x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
        return x, y
    except ValueError:
        pass


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
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = y_pred.max(1)
        for y_p, y_t in zip(predicted, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, )
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
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x_temp = []
        for char in input_string:
            x_temp.append(vocab.get(char, vocab['unk']))
        x.append(x_temp)
        # x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        pro, predicted = result.max(1)
        print('re: ', pro, predicted)
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, predicted[i]+1, pro[i]))  # 打印结果


if __name__ == "__main__":
    # main()
    test_strings = ["fvfeae", "wz你daf", "rqwaeg", "an我www"]
    predict("model.pth", "vocab.json", test_strings)
