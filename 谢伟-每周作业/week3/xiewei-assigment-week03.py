#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import json

'''
构造随机的包含字符'a'的字符串，使用RNN进行多分类，
类别为'a'第一次出此案在字符串中的位置
'''

chars = "abcdefghijklmnopqrstuvwxyz"
vocabulary = {}

# 随机生成一个长度为5的英文字母字符串，随机将其中一个字符替换为字母'a'
def build_data(vocab, sentence_length):
    # 生成随机字符串
    x = [np.random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 检查是否有 'a'，并记录第一个 'a' 的位置
    index = next((i for i, char in enumerate(x) if char == 'a'), -1)
    
    # 如果没有 'a'，随机选择一个位置设置为 'a'
    if index == -1:
        index = np.random.randint(0, sentence_length)
        x[index] = 'a'
    
    # 转换为索引
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, index

def build_sample(vocab, sentence_length, total_count):
    train_x = []
    train_y = []
    for i in range(total_count):
        x, y = build_data(vocab, sentence_length)
        train_x.append(x)
        train_y.append(y)
    return torch.LongTensor(train_x), torch.LongTensor(train_y)

# 自动创建词表
def init_vocabulary():
    vocabulary = {'pad':0}
    for i in range(len(chars)):
        vocabulary[chars[i]] = i + 1
    vocabulary['unk'] = len(chars) + 1
    return vocabulary

# test for every epoch
def evaluate(model, vocab, sentence_length):
    model.eval()
    test_sample_num = 100
    x, y = build_sample(vocab, sentence_length, test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  
            if y_p.argmax() == int(y_t):
                correct += 1   # 与真实标签进行对比,概率最大的一个维度索引与实际分类做对比
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

class TorchModule(nn.Module):
    def __init__(self, vocab_size, vect_dim, hidden_size):
        '''
        vocab_size: 字符串长度
        input_size: RNN输入维度
        hidden_size：RNN隐藏层维度
        output_size:最终输出的维度
        '''
        super(TorchModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vect_dim, padding_idx=0)
        self.rnn = nn.RNN(vect_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x_embeded = self.embedding(x)
        output, _ = self.rnn(x_embeded)
        output = output[:, -1, :]
        y_pred = self.linear(output)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, axis=-1)
    
def main():
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 5   #样本文本长度
    learning_rate = 0.005 #学习率
    hidden_size = 32       #RNN隐藏层的大小
    # 建立字表
    vocab = init_vocabulary()
    print("vocab:", vocab)
    # 建立模型
    model = TorchModule(len(vocab), char_dim, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_sample(vocab, sentence_length, batch_size) #构造一sentence_length组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #保存模型
    torch.save(model.state_dict(), "my_model.pth")
    # 保存词表
    writer = open("my_vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    hidden_size = 32      #RNN隐藏层的大小
    sentence_length = 5  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModule(len(vocab), char_dim, hidden_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    # 初始化此表索引数组
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式

    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for input_string, res in zip(input_strings, result):
        index = np.argmax(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, index, res[index])) #打印结果

if __name__ == "__main__":
    main()
    test_strings = ["fnvae", "wadfg", "rqwae", "ankwa"]
    predict("my_model.pth", "my_vocab.json", test_strings)


