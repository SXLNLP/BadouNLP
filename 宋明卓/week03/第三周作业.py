import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须放在所有库导入之前
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class TorchNet(nn.Module):
    def __init__(self, vocab, vector_idm):
        super(TorchNet, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_idm, padding_idx=0)
        self.layer = nn.RNN(input_size=20, hidden_size=20, batch_first=True)
        self.Linear = nn.Linear(vector_idm, 7)
        self.loss = F.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        _, h = self.layer(x)
        x = h.squeeze()
        y_pred = self.Linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 创建字符集
def build_vocab():
    chars = 'qwertyuiopasdfghjklzxcvbnm'
    vocab = {'pad': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 创建规律
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    if set('a') & set(x):
        y = x.index('a') + 1
    else:
        y = 0

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 创建数据集
def build_dataset(sample_length, sentence_length, vocab):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    x = np.asarray(X)
    y = np.asarray(Y)
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# 测试代码
def test(model, sample_length, sentence_length, vocab):
    model.eval()
    x_test, y_test = build_dataset(sample_length, sentence_length, vocab)
    # 查看样本数据分布
    unique, count = np.unique(y_test, return_counts=True)
    converted_dict = dict(zip(unique, count))
    converted_dict = {int(w + 1): int(v) for w, v in converted_dict.items()}
    print(f'本次预测集中个样本分布-->{converted_dict}')
    correct, worry = 0, 0
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred = y_pred.argmax(dim=-1)
        for y_p, y_t in zip(y_pred, y_test):
            if y_p == int(y_t):
                correct += 1
            else:
                worry += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (worry + correct)))
    return correct / (worry + correct)


def main():
    # 配置参数
    vector_idm = 20  # 每个字符向量维度
    learning_rate = 0.005  # 学习率
    epoch_num = 20  # 训练轮次
    batch_size = 20  # 每次训练样本
    train_sample = 500  # 每次训练总样本
    sentence_length = 6  # 文本长度
    sample_length = 200  # 测试样本数

    # 建立字符集
    vocab = build_vocab()
    # 建立模型
    model = TorchNet(vocab, vector_idm)
    # 选择优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        loss_watch = []
        for batch_num in range(train_sample // batch_size):
            batch_x, batch_y = build_dataset(batch_size, sentence_length, vocab)
            # 计算loss
            loss = model(batch_x, batch_y)
            # 计算梯度
            loss.backward()
            # 权重更新
            optimizer.step()
            # 权重清理（梯度）
            optimizer.zero_grad()
            # 收取每次梯度
            loss_watch.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch, np.mean(loss_watch)))
        acc = test(model, sample_length, sentence_length, vocab)
        log.append([acc, np.mean(loss_watch)])

    # 模型保存
    torch.save(model.state_dict(), 'model.pth')
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    plt.show()

    # 保存词表
    writer = open('vocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    return


# 预测代码
def predict(model, vocab, input_strings):
    vector_idm = 20
    vocab = json.load(open('vocab.json', 'r', encoding='utf-8'))  # 加载词表
    model = TorchNet(vocab, vector_idm)     # 建立模型
    model.load_state_dict(torch.load('model.pth'))      # 加载模型
    x = []
    for string in input_strings:
        x.append([vocab[char] for char in string])
    model.eval()  # 测试模式
    # 不计算梯度
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))  # 模型预测

    for vec, res in zip(input_strings, result):
        print("输入：%s, 预测类别：%s" % (vec, np.argmax(res)))  # 打印结果




if __name__ == '__main__':
    main()
    test_strings = ["favaee", "wzadfg", "aqwdig", "nfkwww"]
    predict("model.pth", "vocab.json", test_strings)
