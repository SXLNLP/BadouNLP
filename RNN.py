#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的RNN模型，完成一个简单的NLP任务：
1. 构造随机包含'a'的字符串
2. 使用RNN进行多分类
3. 分类类别为'a'第一次出现的位置（0到sentence_length-1）
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, sentence_length)  # 输出层，输出各个位置是a的概率
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
        
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        
        # 取最后一个时刻的输出做分类
        x = x[:, -1, :]  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        
        y_pred = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, sentence_length)
        
        if y is not None:
            return self.loss(y_pred, y)  # 计算损失
        else:
            return y_pred  # 返回预测结果

# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz你我他"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符编号
    return vocab

# 随机生成一个样本，强制包含'a'，并返回a第一次出现的位置
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    
    # 强制让'a'出现在字符串中的某个位置(0~sentence_length-1)
    a_position = random.randint(0, sentence_length-1)
    x[a_position] = "a"
    
    # 将字符转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]
    
    # 标签是a第一次出现的位置
    return x, a_position

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 测试代码，用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % len(y))
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred.data, 1)  # 获取最大概率值的索引
        
        for predicted_item, true_item in zip(predicted, y):  # 对比预测值和真实值
            if predicted_item == true_item:
                correct += 1
            else:
                wrong += 1
    
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

# 训练函数
def main():
    # 配置参数
    epoch_num = 15        # 训练轮数
    batch_size = 32       # 每次训练样本个数
    train_sample = 1000   # 每轮训练总共训练的样本总数
    char_dim = 20         # 每个字的维度
    sentence_length = 6   # 样本文本长度
    learning_rate = 0.01  # 学习率
    
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
        # 确保输入字符串长度固定
        if len(input_string) < sentence_length:
            input_string += 'pad' * (sentence_length - len(input_string))
        x.append([vocab.get(char, vocab['unk']) for char in input_string[:sentence_length]])
    
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        _, predicted = torch.max(result.data, 1)  # 获取最大概率值的索引
    
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测位置：%d, 概率值：%f" % (input_string, predicted[i], torch.softmax(result, dim=1)[i][predicted[i]]))  # 打印结果

if __name__ == "__main__":
    # main()
    
    test_strings = ["fnvfeae", "wza你dfg", "rqwdeqa", "n我kawww"]
    print("\n预测结果：")
    predict("model.pth", "vocab.json", test_strings)
