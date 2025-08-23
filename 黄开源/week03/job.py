import torch
import torch.nn as nn
import random
import numpy as np
import json


# 模型
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # RNN循环神经网络
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        # 全连接层
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        # 损失函数
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 构建词表
def build_vocab():
    chars = "abcdefghijk"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    if "a" in x:
        y = x.index("a")
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 准备样本
def build_dataset(sample_length, vocab, sentence_length):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练模型
def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每轮训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立子表
    vocab = build_vocab()
    # print(vocab)
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 保存每轮的测试结果
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(int(train_sample / batch_size)):
            # 构造一组训练样本
            x, y = build_dataset(batch_size, vocab, sentence_length)
            # 梯度归零
            optim.zero_grad()
            # 计算loss
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
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

# 预测
def predict(model_path,vocab_path,input_strings):
    char_dim = 20
    sentence_length = 10
    vocab = json.load(open(vocab_path,"r",encoding="utf8"))
    model = build_model(vocab,char_dim,sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string]) # 将输入序列化
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i,input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string,torch.argmax(result[i]),result[i]))


if __name__ == "__main__":
    # main()
    test_strings = ["aabbccddee", "aebbccadee", "gkijadfbec", "kijhdefacb"]
    predict("model.pth", "vocab.json", test_strings)
