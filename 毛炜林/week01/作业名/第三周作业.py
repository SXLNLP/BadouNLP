import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json

from pyexpat import model


#1、创建数据集

def build_vocab():
    char = "123456789abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index,char in enumerate(char):
        vocab[char] = index+1

    vocab['unk'] = len(vocab)

    return vocab

def build_data(vocab,length):
    #x = [ random.choice(list(vocab.keys())) for _ in range(length)]
    vocab_chars = [c for c in vocab.keys() if c not in ['pad', 'unk']]
    x = random.sample(vocab_chars,length)
    if 'a' in x:
        y = x.index('a')
    else:
        m = np.random.randint(length)
        x[m] = 'a'
        y = m

    x = [vocab.get(word,vocab['unk']) for word in x]

    return x , y

'''

vocab = build_vocab()
writer = open("vocab11.json", "w", encoding="utf8")
writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
writer.close()
data = build_data(vocab,10)
print(data)
'''

def build_dataest(vocab,total_sample,length):
    X = []
    Y = []
    for i in range(total_sample):
        x,y = build_data(vocab,length)
        X.append(x)
        Y.append(y)

    return torch.LongTensor(X),torch.LongTensor(Y)


def build_testdata(vocab,total_sample,length):
    X = []
    Y = []
    for i in range(total_sample):
        x, y = build_data(vocab, length)
        X.append(x)
        Y.append(y)

    return torch.LongTensor(X), torch.LongTensor(Y)

#2、创建模型  embed层有字符总数  维度数  池化层有样本总长度
class Model(nn.Module):
    def __init__(self,vocab,char_dim,hidden_size,sample_length):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab),char_dim,padding_idx=0)
        self.pool = torch.nn.AvgPool1d(sample_length)
        self.rnn = nn.RNN(
            input_size=char_dim,  # 输入维度：词向量维度
            hidden_size=hidden_size,  # 隐藏层维度
            num_layers=1,  # RNN层数
            batch_first=True  # 输入形状为(batch_size, seq_len, input_size)
        )
        self.fc1 = nn.Linear(hidden_size,sample_length)
        self.act1 = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.act = torch.softmax
        self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(0.2)

    def forward(self,x,y=None):
        x = self.embedding(x)
        out, _ = self.rnn(x)  # out形状: (batch_size, seq_len, hidden_size)
        x = out[:, -1, :] # 取最后一个时间步的隐藏状态 (batch_size, hidden_size) -> (batch_size, sample_length)
        #x = x.permute(0,1)
        x=x.unsqueeze(-1)
        #print(x.shape)
        x = self.bn(x)
        #print(x.shape)
        x = x.squeeze(-1)
        #print(x.shape)
        x = self.drop(x)
        x = self.fc1(x)

        if y is not None:

            return self.loss(x,y)
        else:

            return torch.softmax(x,dim=1)


#3、测试代码

def evaluate(model,vocab,length):
    model.eval()
    x,y = build_dataest(vocab,200,length)
    correct =0
    total = 0

    with torch.no_grad():
        y_pred = model(x)  # 输出形状为(batch_size, sample_length)
        _, predicted = torch.max(y_pred, dim=1)  # 预测的0-based索引
        total += y.size(0)
        correct += (predicted == y).sum().item()  # 比较预测索引与真实索引

    accuracy = correct / total
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


#4、训练模型

def train():
    epochs = 20
    batch_size = 32
    total_sample = 2000
    char_dim = 32
    sample_length = 6
    lerning_rate = 0.001
    hidden_size = 64
    log = []
    vocab = build_vocab()
    model = Model(vocab,char_dim,hidden_size,sample_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=lerning_rate)

    for epoch in range(epochs):
        model.train()
        loss_total = []
        for i in range(total_sample//batch_size):
            x,y = build_dataest(vocab,batch_size,sample_length)
            optimizer.zero_grad()
            loss = model(x,y)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(loss_total)))
        acc = evaluate(model, vocab, sample_length)  # 测试本轮模型结果
        log.append([acc, np.mean(loss_total)])

        # 保存模型
    torch.save(model.state_dict(), "modelfenglei.pth")
        # 保存词表

    with open("vocab11.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return

#4、预测
def predict(model_path, vocab_path, x_test, y_test):
    char_dim = 32
    sample_length = 6
    hidden_size = 64
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    #vocab = json.load(open(vocab_path,"r",encoding="utf8"))
    model = Model(vocab,char_dim,hidden_size,sample_length)
    model.load_state_dict(torch.load(model_path))


    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        _, predicted = torch.max(y_pred, dim=1)
        for pred,true in zip(predicted,y_test):
            print(f"预测位置：{pred.item() + 1}\t真实位置：{true.item() + 1}")



if __name__ == '__main__':
    train()
    vocab = build_vocab()
    x,y = build_testdata(vocab,200,6)
    predict("modelfenglei.pth","vocab11.json",x,y)






