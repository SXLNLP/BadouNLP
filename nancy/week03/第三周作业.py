#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于RNN的网络实现
构造随机包含a的字符串，类别为a第一次出现在字符串中的位置
"""

class RNNModel(nn.Module):
    def __init__(self, char_dim, hidden_size, sentence_length, vocab):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim, padding_idx=0)  # embedding层
        self.layer = nn.RNN(
            input_size=char_dim,
            hidden_size=hidden_size,
            bias=False,
            batch_first=True
        )
        # 线性层，输出维度为句子长度（可能的位置数）
        self.classify = nn.Linear(hidden_size, sentence_length)
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        # RNN处理
        output, hidden = self.layer(x)
        last_output = output[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(last_output)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果的概率分布

# 字符集
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字，确保包含一个'a'字符
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 随机选择一个位置插入'a'
    a_position = random.randint(0, sentence_length-1)
    x[a_position] = 'a'
    # 将字符转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, a_position

# 建立数据集
# 输入需要的样本数量
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, hidden_size, sentence_length):
    model = RNNModel(char_dim, hidden_size, sentence_length, vocab)
    return model

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % len(y))
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            pred_pos = torch.argmax(y_p).item()  # 获取预测概率最大的位置
            if pred_pos == y_t.item():  # 比较预测位置和真实位置
                correct += 1  # 位置预测正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    # 配置参数
    epoch_num = 20        #训练轮数
    batch_size = 30       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    hidden_size = 40  # RNN隐藏层维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.002  # 学习率
    
    
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, hidden_size, sentence_length)
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
    torch.save(model.state_dict(), "homework3.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    hidden_size = 40  # RNN隐藏层维度
    sentence_length = 6  # 样本文本长度
    
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, hidden_size, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    
    # 将输入字符串转换为模型输入
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        pred_pos = torch.argmax(result[i]).item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, pred_pos, result[i][pred_pos])) #打印结果

if __name__ == "__main__":
    main()
    
    # 训练完成后，用以下代码测试模型
    test_strings = ["adfghi", "dafghi", "dfaghi", "dfgahi", "dfghai", "dfghia"]
    predict("homework3.pth", "vocab.json", test_strings) 
