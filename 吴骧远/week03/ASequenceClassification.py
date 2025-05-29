# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
"""
基于pytorch的RNN网络编写
实现RNN完成多分类任务.字符串为定长6
预测字符'a'第一次出现在字符串中的位置（0-5）
"""
class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=32, num_classes=6):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)  # RNN层
        self.classify = nn.Linear(hidden_size, num_classes)  # 线性层，输出6个类别（位置0-5）
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数用于多分类
    def forward(self, x, y=None):
        # x: (batch_size, sen_len)
        x = self.embedding(x)  # (batch_size, sen_len, vector_dim)
        # RNN处理序列
        rnn_out, hidden = self.rnn(x)  # rnn_out: (batch_size, sen_len, hidden_size)
        # 使用最后一个时间步的输出进行分类
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        # 分类
        logits = self.classify(last_output)  # (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)  # 计算损失
        else:
            return torch.softmax(logits, dim=1)  # 返回概率分布
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    """
    构造随机包含'a'的长度为6的字符串
    返回字符串和'a'第一次出现的位置
    """
    # 可选字符（除了'a'）
    other_chars = [char for char in vocab.keys() if char not in ['pad', 'unk', 'a']]
    # 随机选择'a'第一次出现的位置（0-5）
    a_position = random.randint(0, sentence_length - 1)
    # 构造字符串
    x = []
    for i in range(sentence_length):
        if i == a_position:
            x.append('a')  # 在指定位置放置'a'
        elif i > a_position and random.random() < 0.3:  # 30%概率在'a'后面再次出现'a'
            x.append(random.choice(['a'] + other_chars))  # 可能再次出现'a'或其他字符
        else:
            x.append(random.choice(other_chars))  # 选择其他字符
    # 找到'a'第一次出现的位置
    first_a_pos = x.index('a')
    # 将字符转换为索引
    x_indices = [vocab.get(char, vocab['unk']) for char in x]
    return x_indices, first_a_pos

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 标签是位置索引（0-5）
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = RNNModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    # 统计各位置的样本数量
    position_counts = [0] * 6
    for pos in y:
        position_counts[pos.item()] += 1
    print("测试集中各位置的样本数量：")
    for i, count in enumerate(position_counts):
        print(f"位置{i}: {count}个样本")
    correct = 0
    total = 0
    with torch.no_grad():
        y_pred = model(x)  # 获取概率分布
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        for pred, true in zip(predicted_classes, y):
            if pred == true:
                correct += 1
            total += 1
    accuracy = correct / total
    print(f"正确预测个数：{correct}, 总样本数：{total}, 准确率：{accuracy:.4f}")
    return accuracy
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 32  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率
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
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.6f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
def predict(model_path, vocab_path, input_strings):
    char_dim = 32
    sentence_length = 6
    # 加载词汇表
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    # 处理输入字符串
    x = []
    for input_string in input_strings:
        # 确保字符串长度为6
        if len(input_string) != 6:
            print(f"警告：输入字符串 '{input_string}' 长度不为6，将进行截断或填充")
            input_string = input_string[:6].ljust(6, 'z')  # 截断或用'z'填充
        indices = [vocab.get(char, vocab['unk']) for char in input_string]
        x.append(indices)

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))  # 获取概率分布
        predicted_positions = torch.argmax(result, dim=1)  # 获取预测位置

    # 打印结果
    for i, input_string in enumerate(input_strings):
        actual_pos = input_string.find('a') if 'a' in input_string else -1
        predicted_pos = predicted_positions[i].item()
        print(f"输入：{input_string}")
        print(f"实际'a'的位置：{actual_pos}")
        print(f"预测'a'的位置：{predicted_pos}")
        print("各位置概率分布：", [f"{j}:{result[i][j].item():.6f}" for j in range(6)])
        print("-" * 40)

if __name__ == "__main__":
    # 训练模型
    main()
    # 测试字符串
    test_strings = ["fnafee", "wa你dfg", "rqcdag", "nmkwwa","adfre4","jkbafr5"]
    print("\n" + "=" * 50)
    predict("rnn_model.pth", "vocab.json", test_strings)
