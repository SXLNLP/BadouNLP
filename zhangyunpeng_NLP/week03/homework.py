import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于PyTorch的RNN网络实现
判断字符'a'首次出现在字符串中的位置
"""

class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size, hidden_size):
        super(RNNModel, self).__init__()
        self.sentence_length = sentence_length
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.GRU(vector_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, sentence_length + 1)  # 0到sentence_length共sentence_length+1个类别
        self.loss = nn.CrossEntropyLoss()  # 多分类损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)  # (batch_size, sen_len, hidden_size)
        output = output[:, -1, :]  # 取最后时刻的隐藏状态 (batch_size, hidden_size)
        y_pred = self.classifier(output)  # (batch_size, sentence_length + 1)
        
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return torch.softmax(y_pred, dim=1)  # 返回概率分布

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    return vocab

def build_sample(vocab, sentence_length):
    # 随机决定字符'a'首次出现的位置(0到sentence_length-1)，sentence_length表示没有'a'
    a_position = random.randint(0, sentence_length)
    
    if a_position < sentence_length:  # 字符串中包含'a'
        # 生成不包含'a'的字符列表
        other_chars = [char for char in vocab.keys() if char != 'a' and char != 'pad']
        # 初始化字符串列表，默认全是其他字符
        x = [random.choice(other_chars) for _ in range(sentence_length)]
        # 在指定位置插入'a'
        x[a_position] = 'a'
    else:  # 字符串中不包含'a'
        other_chars = [char for char in vocab.keys() if char != 'a' and char != 'pad']
        x = [random.choice(other_chars) for _ in range(sentence_length)]
    
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, a_position

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab_size, char_dim, sentence_length, hidden_size):
    model = RNNModel(char_dim, sentence_length, vocab_size, hidden_size)
    return model

def evaluate(model, vocab, sentence_length, sample_length=200):
    model.eval()
    x, y = build_dataset(sample_length, vocab, sentence_length)
    correct = 0
    total = 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    accuracy = correct / total
    print(f"评估结果: 准确率 {accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    epoch_num = 10
    batch_size = 32
    train_sample = 1000
    char_dim = 20
    sentence_length = 8
    hidden_size = 64
    learning_rate = 0.001
    
    # 建立字表
    vocab = build_vocab()
    vocab_size = len(vocab)
    
    # 建立模型
    model = build_model(vocab_size, char_dim, sentence_length, hidden_size)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        
        print(f"Epoch {epoch+1}/{epoch_num}, 平均损失: {np.mean(watch_loss):.4f}")
        evaluate(model, vocab, sentence_length)
    
    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    with open("rnn_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # 测试预测
    test_strings = ["abcdefgh", "bcdaefgh", "bcdefgha", "bcdefghx", "a"]
    predict("rnn_model.pth", "rnn_vocab.json", test_strings, char_dim, sentence_length, hidden_size)

def predict(model_path, vocab_path, input_strings, char_dim, sentence_length, hidden_size):
    # 加载词表
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # 创建模型
    model = build_model(len(vocab), char_dim, sentence_length, hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 处理输入
    inputs = []
    for s in input_strings:
        # 截断或填充到固定长度
        s = s[:sentence_length].ljust(sentence_length, 'pad')
        inputs.append([vocab.get(c, vocab['unk']) for c in s])
    
    # 预测
    with torch.no_grad():
        inputs = torch.LongTensor(inputs)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    
    # 输出结果
    for i, s in enumerate(input_strings):
        position = predicted[i].item()
        print(f"输入: '{s}', 预测'a'首次出现位置: {position if position < sentence_length else '未出现'}")

if __name__ == "__main__":
    main()    
