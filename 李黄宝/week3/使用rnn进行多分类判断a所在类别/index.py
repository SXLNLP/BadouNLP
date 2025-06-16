#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from collections import Counter

# 
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz!@#$%^&*"  # 字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=64):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)  # RNN层
        self.classify = nn.Linear(hidden_size, sentence_length + 1)  # 线性层，输出类别数=字符串长度+1
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_output, _ = self.rnn(x) 
        last_output = rnn_output[:, -1, :]
        y_pred = self.classify(last_output)
        
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机生成字符串
    chars = list(vocab.keys())
    chars.remove('pad')
    chars.remove('unk')
    
    # 生成随机字符串
    x = [random.choice(chars) for _ in range(sentence_length)]
    if random.random() < 0.8:
        a_position = random.randint(0, sentence_length-1)
        if a_position < len(x):
            x[a_position] = 'a'
        
        y = a_position + 1
    else:
        y = 0
    
    x = [vocab.get(char, vocab['unk']) for char in x]
    
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    position_counts = [0] * (sentence_length + 1)  # 统计每个位置的数量
    
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
        position_counts[y] += 1
    
    print(f"数据集类别分布: {position_counts}")
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 评估模型
def evaluate(model, vocab, sentence_length):
    model.eval()
    test_size = 200
    x, y = build_dataset(test_size, vocab, sentence_length)
    
    class_counts = Counter(y.numpy())
    print(f"测试集类别分布: {sorted(class_counts.items())}")
    
    correct = 0
    with torch.no_grad():
        y_pred = model(x) 
        predictions = torch.argmax(y_pred, dim=1) 
        
        # 计算正确率
        correct = (predictions == y).sum().item()
        accuracy = correct / test_size
    
    print(f"正确预测个数: {correct}, 正确率: {accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    epoch_num = 20          # 训练轮数
    batch_size = 32         # 每次训练样本个数
    train_sample = 2000     # 每轮训练总共训练的样本总数
    char_dim = 32           # 每个字的维度
    sentence_length = 6     # 样本文本长度
    learning_rate = 0.001   # 学习率
    
    # 建立字表
    vocab = build_vocab()
    print(f"字符表大小: {len(vocab)}")
    
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        batch_count = train_sample // batch_size
        
        for batch in range(batch_count):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / batch_count
        print(f"=========\n第{epoch+1}轮平均loss: {avg_loss:.4f}")
        
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, avg_loss])
    
    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    
    # 保存词表
    with open("rnn_vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 32
    sentence_length = 6
    
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    
    model = build_model(vocab, char_dim, sentence_length)
    
    model.load_state_dict(torch.load(model_path))
    x = []
    for s in input_strings:
        if len(s) > sentence_length:
            s = s[:sentence_length]
        elif len(s) < sentence_length:
            s = s + ' ' * (sentence_length - len(s))
        
        seq = [vocab.get(char, vocab['unk']) for char in s]
        x.append(seq)
    
    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.LongTensor(x))
        probabilities = torch.softmax(y_pred, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # 打印结果
    print("\n预测结果:")
    for i, s in enumerate(input_strings):
        pred_class = predictions[i].item()
        max_prob = torch.max(probabilities[i]).item()
        
        if pred_class == 0:
            result = "无a字母类别"
        else:
            result = f"{pred_class}"
        
        print("输入:%s , 预测类别:%s , 类别概率:%s" %(s, result, max_prob))

if __name__ == "__main__":
    main()
    
    # 测试字符串
    test_strings = [
        "bcdefg",
        "a12345",
        "xayz!@",
        "12a345",
        "123a45",
        "1234a5",
        "12345a",
        "aabcde",
        "xyzabc",
        "!@#$%a",
        "noletter", 
        "randomstr"
    ]
    
    predict("rnn_model.pth", "rnn_vocab.json", test_strings)
