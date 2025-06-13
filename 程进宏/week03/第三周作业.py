# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size=128, batch_first=True)

        self.classify = nn.Linear(128, num_classes)  #线性层

        self.sentence_length = sentence_length
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len, vector_dim)
        output, hidden = self.rnn(x)  # 获取RNN输出
        x = output[:, -1, :]  # 取最后一个时间步的输出

        # 通过分类层
        x = self.classify(x)  # (batch_size, num_classes)

        if y is not None:
            # 计算交叉熵损失
            return self.loss(x, y.long())
        else:
            return x  # 返回原始logits

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "bcdefghijklmnopqrstuvwxyz0123456789"  # 不包含'a'
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1

    # 单独添加'a'
    vocab['a'] = len(vocab) + 1
    vocab['unk'] = len(vocab) + 1
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    # 获取所有非'a'字符
    non_a_chars = [char for char in vocab.keys() if char != 'a' and char != 'pad' and char != 'unk']

    # 生成非'a'字符
    chars = [random.choice(non_a_chars) for _ in range(sentence_length - 1)]

    # 随机选择位置插入'a'
    pos = random.randint(0, sentence_length - 1)
    chars.insert(pos, 'a')

    # 转换为索引序列
    x = [vocab.get(char, vocab['unk']) for char in chars]
    return x, pos

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
def build_model(vocab, char_dim, sentence_length, num_classes):
    return RNNModel(char_dim, sentence_length, vocab, num_classes)

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length, num_classes):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)

    # 统计每个位置的样本数
    pos_counts = [0] * num_classes
    for pos in y:
        pos_counts[pos.item()] += 1
    print(f"样本位置分布: {pos_counts}")

    correct = 0
    total = len(y)
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output, 1)
        correct = (predicted == y).sum().item()

    accuracy = correct / total
    print(f"正确预测个数: {correct}/{total}, 正确率: {accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20          # 增加训练轮数
    batch_size = 64         # 增大批大小
    train_sample = 2000     # 增加训练样本量
    char_dim = 32           # 增加字符维度
    sentence_length = 6     # 字符串长度固定为6
    num_classes = sentence_length  # 类别数为6（位置0-5）
    learning_rate = 0.001   # 学习率
    # 建立字表
    vocab = build_vocab()
    print("词汇表大小:", len(vocab))
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0  # 初始化最佳准确率
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        total_batches = int(train_sample / batch_size)
        for batch in range(total_batches):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / total_batches
        print(f"Epoch [{epoch + 1}/{epoch_num}], 平均Loss: {avg_loss:.4f}")
        # 每轮评估
        acc = evaluate(model, vocab, sentence_length, num_classes)
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "position_model.pth")
            print("保存最佳模型")
    # 保存最终模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 32  # 每个字的维度 与训练时一致
    sentence_length = 6 # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))                       #加载字符表
    model = RNNModel(char_dim, sentence_length, vocab, num_classes=sentence_length) #建立模型
    model.load_state_dict(torch.load(model_path))                                   #加载训练好的权重
    # 反转词汇表（索引到字符）
    idx_to_char = {idx: char for char, idx in vocab.items()}
    x = []
    valid_strings = []  # 有效的输入字符串
    for input_string in input_strings:
        # 检查长度
        if len(input_string) != sentence_length:
            continue
        # 检查'a'的数量
        a_count = input_string.count('a')
        if a_count != 1:
            continue
        # 将输入序列化
        sequence = []
        for char in input_string:
            if char in vocab:
                sequence.append(vocab[char])
            else:
                sequence.append(vocab['unk'])
        x.append(sequence)
        valid_strings.append(input_string)
    if not x:
        print("没有有效的输入字符串")
        return
    x = torch.LongTensor(x)
    model.eval()
    with torch.no_grad():
        result = model(x)  # 获取模型输出
        probabilities = torch.softmax(result, dim=1)  # 转换为概率分布
        predicted_positions = torch.argmax(probabilities, dim=1)  # 获取预测位置
    for i, input_string in enumerate(valid_strings):
        pred_pos = predicted_positions[i].item()
        actual_pos = input_string.index('a')
        confidence = probabilities[i][pred_pos].item()  # 预测位置的置信度
        # 显示结果
        result_char = '正确' if pred_pos == actual_pos else '错误'
        print(
            f"输入: {input_string}, 预测位置: {pred_pos}, 实际位置: {actual_pos}, {result_char}, 置信度: {confidence:.4f}")


if __name__ == "__main__":
    main()
    # test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    # 测试样本
    test_strings = [
        "abcde1",  # 'a'在位置0
        "0abcde",  # 'a'在位置1
        "01abcd",  # 'a'在位置2
        "012abc",  # 'a'在位置3
        "0123ab",  # 'a'在位置4
        "01234a",  # 'a'在位置5
        "a0b0c0"  # 'a'在位置0
    ]

    predict("position_model.pth", "vocab.json", test_strings)
