#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层

        # 使用RNN + 交叉熵实现
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.CrossEntropyLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        final_out = rnn_out[:, -1, :]
        y_pred = self.classify(final_out)  # (batch_size, sentence_length + 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  #使用交叉熵计算损失函数，所以需要使用softmax 归1

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他adefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    """
    构造随机包含a的字符串, 类别为a第一次出现在字符串中的位置。
    :param vocab:
    :param sentence_length:
    :return:
    """
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if 'a' in x:
        y = x.index('a')  # a第一次出现的位置
    else:
        y = sentence_length  # 如果没有a，则为负样本，类别为sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
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
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)

    # 统计正负样本
    positive_num = (y != sample_length).sum().item()
    negative_num = (y == sample_length).sum().item()
    print("本次预测集中共有%d个正样本，%d个负样本" % (positive_num, negative_num))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred_probs = model(x)  # 获取概率分布
        y_pred = torch.argmax(y_pred_probs, dim=1)  # 获取预测的类别

        correct = (y_pred == y).sum().item()
        wrong = len(y) - correct

    accuracy = correct / (correct + wrong)
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
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        result = torch.argmax(result, dim=1)
    for i, input_string in enumerate(input_strings):
        if len(input_string) == sentence_length:
            pos = result[i].item()
            if pos < sentence_length:
                print(f"输入: {input_string}, 预测'a'位置: {pos}")
            else:
                print(f"输入: {input_string}, 预测不含'a'")



if __name__ == "__main__":
    # main()
    test_strings = ["fnvfae", "wza你df", "arqwde", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
