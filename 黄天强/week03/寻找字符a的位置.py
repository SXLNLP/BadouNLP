import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json

"""
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # 第一个维度为bs
        self.act = nn.ReLU()
        self.fc = nn.Linear(vector_dim, sentence_length + 1)  # +1表示不包含a的情况

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """当输入真实标签，返回loss值；无真实标签，返回预测值"""
        x = self.embedding(x)
        output, hidden = self.rnn(x)

        y_pred = self.fc(self.act(output[:, -1, :]))  # torch.Size([20, 7])
        # y_pred = self.fc(self.act(hidden[-1]))  # torch.Size([20, 7])

        if y is not None:
            return self.loss(y_pred, y)  # 损失
        else:
            return torch.softmax(y_pred, dim=1)  # 预测结果


def build_vocab():
    """
    建立字表。
    说明：字符集随便挑了一些字，实际上还可以扩充，为每个字生成一个标号，abc -> [1,2,3]
    例：{'pad': 0, 'a':1, 'b':2, 'c':3 , ..., 'unk': 27}
    """
    # chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    chars = "abcdefghijkl"  # 字符集（缩短加速训练）
    vocab = {'pad': 0}  # 填充符
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符
    return vocab


def build_sample(vocab, sentence_length):
    """
    随机生成一个样本。
    从所有字中选取sentence_length个字，标签为'a'在字符串中的位置，没有'a'为负样本。
    """
    # 1
    # x = random.sample(list(vocab.values()), sentence_length)  # 不重复采样
    #
    # print(x)
    # if 'a' in x:  # 正样本
    #     y = x.index('a')
    # else:
    #     y = sentence_length
    # x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    # return x, y

    # 2
    x = random.sample(list(vocab.values()), sentence_length)  # 不重复采样
    if 1 in x:  # 正样本
        y = x.index(1)
    else:
        y = sentence_length
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    """建立数据集"""
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.tensor(dataset_x), torch.tensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    """建立模型"""
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length):
    """测试每轮模型的准确率"""
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 200个样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("共有%d个样本\n正确预测个数：%d，正确率：%f" % (len(x), correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch = 10  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 32  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = build_vocab()

    model = build_model(vocab, char_dim, sentence_length)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log = []

    for epoch in range(epoch):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss).item()))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    """使用训练好的模型做预测"""
    char_dim = 32  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        index = torch.argmax(result[i]).item()
        print("%s  （'a'的位置: %d, 概率值: %f）" % (input_string, index, result[i][index].item()))  # 打印结果
        print(' ' * index + '↑' + '×' * (index >= sentence_length))


if __name__ == "__main__":
    main()

    print("=========")
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb", "bcdefghigk"]
    predict("model.pth", "vocab.json", test_strings)
