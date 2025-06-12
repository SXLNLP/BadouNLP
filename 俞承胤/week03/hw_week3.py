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
    def __init__(self, vector_dim, hidden_size,vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        #print("vector_dim",vector_dim)
        self.rnn_layer = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)
        self.linear_layer = nn.Linear(5, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        #print("x1",x)
        x = self.embedding(x)                   #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, hidden = self.rnn_layer(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*20 20*1 -> 3*1
        y_pred = output[:, -1, :]
        y_pred = self.linear_layer(y_pred)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

def build_vocab():
    vocab = {"pad": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10,
             "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20,
             "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26, "unk": 27}
    return vocab

def build_sample():
    x = random.sample(range(2, 27), 5)
    idx = random.sample(range(0, 4), 1)
    idx = idx[0]
    x[idx] = 1

    y = [0, 0, 0, 0, 0]
    y[idx] = 1

    return x, y

def build_dataset(sample_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample()
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(100)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 8  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(char_dim, sentence_length, vocab)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size)  # 构造一组训练样本
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
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

def predict(model_path, vocab_path, input_strings):
    char_dim = 8  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModel(char_dim, sentence_length, vocab)
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    print(x)
    print(result)


if __name__ == "__main__":
    #main()
    test_strings = ["abcde", "djgad", "rqaeg", "daoew"]
    predict("model.pth", "vocab.json", test_strings)
