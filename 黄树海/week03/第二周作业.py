#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务: 句子分类
判断文本序列中目标字符（你or我or他）出现在文本的的第几个位置就是第几分类，输入文本长度为6，输出为第几分类

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=vector_dim,
                           hidden_size=vector_dim,
                           bias=False,
                           batch_first=True)
        self.classify = nn.Linear(vector_dim, 6)  # 输出 6 类别（0~5）
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, hidden = self.rnn(x)  # rnn_out: (batch_size, sen_len, vector_dim)

        # 取最后一个时间步的输出作为句子表示（或可取所有时间步最大池化）
        x = output[:, -1, :]  # (batch_size, vector_dim)
        # x = torch.max(output, dim=1)[0]   # 效果不好

        y_pred = self.classify(x)  # (batch_size, 6)

        if y is not None:
            return self.loss(y_pred, y.long().squeeze())  # 计算 loss
        else:
            return torch.softmax(y_pred, dim=1)  # 返回预测概率分布

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = None
    for i, char in enumerate(x):
        if char in "你我他":
            y = i
            break
    if y is None:
        return build_sample(vocab, sentence_length)  # 递归重新生成
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    while len(dataset_x) < sample_length:
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted_classes = torch.argmax(y_pred, dim=1)
        correct = (predicted_classes == y).sum().item()
    accuracy = correct / 200
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
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
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model_2.pth")
    # 保存词表
    writer = open("vocab_2.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        predicted_class = torch.argmax(result[i]).item()
        prob = float(result[i][predicted_class])
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, predicted_class+1, prob))   # 预测类别转化为范围：1~6


if __name__ == "__main__":
    main()
    test_strings = ["啊他他%1q","GFD是谁你","fn他fee", "w1p你fg", "你qwdea", "kwww09","ab34我9"]
    predict("model_2.pth", "vocab_2.json", test_strings)
