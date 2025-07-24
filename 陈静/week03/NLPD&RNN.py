import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的网络编写
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
"""

class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(RNNModel, self).__init__()
        self.vector_dim = vector_dim
        self.sentence_length = sentence_length
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # 嵌入层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, sentence_length)  # 分类层，输出维度为句子长度（位置数）
        self.activation = torch.softmax  # 多分类使用softmax激活
        
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_output, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim)
        # 取最后一个时间步的输出作为整个序列的表示
        output = rnn_output[:, -1, :]  # (batch_size, vector_dim)
        y_pred = self.classify(output)  # (batch_size, sentence_length)
        y_pred = self.activation(y_pred, dim=1)  # 应用softmax到每个样本
        
        if y is not None:
            loss = nn.functional.cross_entropy(y_pred, y)  # 多分类使用交叉熵损失
            return loss
        else:
            return y_pred  # 输出预测的位置概率分布

# 构建字符集和映射表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 包含a的字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字符对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符标记
    return vocab

# 生成单个样本
def build_sample(vocab, sentence_length):
    # 确保字符串中包含至少一个'a'
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 随机确定第一个'a'的位置
    first_a_pos = random.randint(0, sentence_length - 1)
    x[first_a_pos] = 'a'  # 在确定位置放置'a'
    
    # 将字符转换为编号
    x = [vocab.get(char, vocab['unk']) for char in x]
    y = first_a_pos  # 标签为第一个'a'的位置
    return x, y

# 生成数据集
def build_dataset(sample_count, vocab, sentence_length):
    X, Y = [], []
    for i in range(sample_count):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

# 训练模型
def train(model, vocab, sentence_length, vector_dim, epoch=100, batch_size=32, lr=0.001):
    # 构建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 构建数据集
    train_X, train_Y = build_dataset(10000, vocab, sentence_length)
    
    # 训练过程
    for e in range(epoch):
        model.train()
        total_loss = 0
        # 批量训练
        for i in range(0, len(train_X), batch_size):
            batch_X = train_X[i:i+batch_size]
            batch_Y = train_Y[i:i+batch_size]
            # 前向传播
            loss = model(batch_X, batch_Y)
            total_loss += loss.item()
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 打印训练信息
        print(f"Epoch {e+1}/{epoch}, Loss: {total_loss / (len(train_X)/batch_size)}")
        
        # 每10轮进行一次验证
        if (e+1) % 10 == 0:
            accuracy = evaluate(model, vocab, sentence_length, 1000)
            print(f"Epoch {e+1}/{epoch}, Accuracy: {accuracy}")

# 评估模型
def evaluate(model, vocab, sentence_length, sample_count=100):
    model.eval()
    X, Y = build_dataset(sample_count, vocab, sentence_length)
    correct = 0
    with torch.no_grad():
        y_pred = model(X)
        # 找到概率最大的位置
        pred_pos = torch.argmax(y_pred, dim=1)
        correct = (pred_pos == Y).sum().item()
    return correct / sample_count

# 预测单个样本
def predict(model, vocab, sentence_length, input_str):
    model.eval()
    # 转换输入字符串为编号
    x = [vocab.get(char, vocab['unk']) for char in input_str]
    # 补全或截断到固定长度
    if len(x) < sentence_length:
        x += [vocab['pad']] * (sentence_length - len(x))
    else:
        x = x[:sentence_length]
    x = torch.LongTensor([x])
    
    with torch.no_grad():
        y_pred = model(x)
        pred_pos = torch.argmax(y_pred, dim=1).item()
    
    return pred_pos

# 主函数
def main():
    # 配置参数
    vector_dim = 5  # 五维向量
    sentence_length = 10  # 字符串长度
    vocab = build_vocab()
    
    # 构建模型
    model = RNNModel(vector_dim, sentence_length, vocab)
    
    # 训练模型
    train(model, vocab, sentence_length, vector_dim)
    
    # 测试预测功能
    test_str = "bcdfga"
    pred_pos = predict(model, vocab, sentence_length, test_str)
    print(f"测试字符串: {test_str}")
    print(f"预测第一个'a'的位置: {pred_pos}")
    print(f"实际第一个'a'的位置: {test_str.index('a')}")

if __name__ == "__main__":
    main()
