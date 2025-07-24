# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from collections import Counter

"""

基于pytorch的RNN网络实现
任务：预测字符串中第一个'a'出现的位置（多分类问题）
类别说明：位置0-5（字符串位置），没有'a'则为6（共7个类别）

"""


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(RNNModel, self).__init__()
        # 嵌入层：将字符索引转换为向量
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)

        # RNN层：处理序列信息
        self.rnn = nn.RNN(input_size=vector_dim,
                          hidden_size=vector_dim * 2,  # 隐藏层大小是输入向量的两倍
                          batch_first=True)  # 输入格式为(batch, seq, feature)

        # 分类层：输出位置概率（7个类别）
        self.classify = nn.Linear(vector_dim * 2, sentence_length + 1)

        # 损失函数：交叉熵（适合多分类）
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 嵌入层处理
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)

        # RNN处理序列
        output, hidden = self.rnn(x)  # output: (batch, seq_len, hidden_size)
        # hidden: (1, batch, hidden_size)

        # 提取每个序列最后一个时间步的输出（包含序列的整体信息）
        last_output = output[:, -1, :]  # (batch, hidden_size)

        # 分类层输出7个类别的分数
        y_pred = self.classify(last_output)  # (batch, 7)

        if y is not None:
            # 计算交叉熵损失
            return self.loss(y_pred, y.long().squeeze())  # 需要将标签转换为长整型并去除多余维度
        else:
            return y_pred  # 输出预测结果


# 字符集（这里主要使用小写字母）
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 英文字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字符对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符标记
    return vocab


# 生成一个样本
def build_sample(vocab, sentence_length):
    # 生成随机字符串（长度为sentence_length）
    chars = list(vocab.keys())
    chars.remove('pad')  # 移除填充标记
    chars.remove('unk')  # 移除未知标记

    # 创建初始字符串（至少包含一个字符）
    base_chars = [c for c in chars if c != 'a']  # 排除'a'的字符
    x = [random.choice(base_chars) for _ in range(sentence_length)]

    # 随机插入1-3个'a'字符
    for _ in range(random.randint(1, 3)):
        pos = random.randint(0, sentence_length - 1)
        x[pos] = 'a'

    # 查找第一个'a'出现的位置
    try:
        first_a = x.index('a')  # 第一个'a'的位置
        y = first_a
    except ValueError:
        y = sentence_length  # 没有'a'的情况（类别6）

    # 将字符转换为索引
    x = [vocab.get(char, vocab['unk']) for char in x]

    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 使用单个整数表示类别（0-6）

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 测试模型性能
def evaluate(model, vocab, sentence_length, sample_size=200):
    model.eval()
    x, y_true = build_dataset(sample_size, vocab, sentence_length)

    # 统计类别分布
    class_counts = Counter(y_true.numpy())
    print("测试集类别分布:")
    for cls in range(sentence_length + 1):
        count = class_counts.get(cls, 0)
        desc = f"位置 {cls}" if cls < sentence_length else "没有'a'"
        print(f"  {desc}: {count} 个样本")

    correct = 0
    total = 0

    # 预测并统计结果
    with torch.no_grad():
        y_pred = model(x)  # 获取预测分数
        _, predicted_classes = torch.max(y_pred, dim=1)  # 获取预测类别

        # 计算准确率
        correct = (predicted_classes == y_true).sum().item()
        total = y_true.size(0)
        accuracy = correct / total

        # 打印混淆位置（常见错误）
        error_positions = []
        for true, pred in zip(y_true, predicted_classes):
            if true != pred:
                error_positions.append((true.item(), pred.item()))

        print(f"测试准确率: {accuracy:.4f} ({correct}/{total})")

        if error_positions:
            # 统计常见错误
            error_counter = Counter(error_positions)
            print("\n常见错误分析（真实位置 -> 预测位置）:")
            for (true, pred), count in error_counter.most_common(5):
                true_desc = f"位置 {true}" if true < sentence_length else "没有'a'"
                pred_desc = f"位置 {pred}" if pred < sentence_length else "没有'a'"
                print(f"  {true_desc} -> {pred_desc}: {count}次")

    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每批次样本数
    train_sample = 800  # 每轮训练样本总数
    char_dim = 64  # 字符向量的维度
    sentence_length = 6  # 字符串长度
    num_classes = sentence_length + 1  # 类别数（位置0-5 + 没有'a' = 7）
    learning_rate = 0.001  # 学习率

    print("=" * 50)
    print(f"任务: 预测字符串中第一个'a'出现的位置")
    print(f"字符串长度: {sentence_length}")
    print(f"类别数: {num_classes} (0-5位置 + 没有'a')")
    print("=" * 50)

    # 建立字符表
    vocab = build_vocab()
    print(f"字符表大小: {len(vocab)}")

    # 建立模型
    model = RNNModel(char_dim, sentence_length, vocab)
    print("模型结构:")
    print(model)

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    print("\n开始训练...")
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0.0

        # 构建训练数据
        x, y = build_dataset(train_sample, vocab, sentence_length)

        # 分批训练
        for i in range(0, train_sample, batch_size):
            # 获取当前批次
            batch_end = i + batch_size
            batch_x = x[i:batch_end]
            batch_y = y[i:batch_end]

            # 前向传播和计算损失
            optimizer.zero_grad()
            loss = model(batch_x, batch_y)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        # 计算平均损失
        epoch_loss /= train_sample

        # 每轮结束评估
        print(f"\nEpoch {epoch + 1}/{epoch_num}, 平均损失: {epoch_loss:.4f}")
        acc = evaluate(model, vocab, sentence_length, sample_size=200)

        # 简单的学习率衰减
        if epoch > 5 and epoch_loss < 0.5:
            learning_rate = max(learning_rate * 0.9, 0.0001)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"更新学习率: {learning_rate:.6f}")

    print("\n训练完成!")

    # 保存模型
    torch.save(model.state_dict(), "position_predictor.pth")

    # 保存字符表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("模型和字符表已保存")


# 使用模型进行预测
def predict(model_path, vocab_path, input_strings):
    # 加载字符表
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)

    # 模型参数
    char_dim = 64
    sentence_length = 6

    # 构建模型并加载权重
    model = RNNModel(char_dim, sentence_length, vocab)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 准备输入数据
    x = []
    for s in input_strings:
        # 截取或填充字符串到固定长度
        s = s[:sentence_length].ljust(sentence_length, random.choice(list(vocab.keys())))
        # 转换字符为索引
        indices = [vocab.get(c, vocab['unk']) for c in s]
        x.append(indices)

    # 转换为张量
    x_tensor = torch.LongTensor(x)

    # 预测
    with torch.no_grad():
        y_pred = model(x_tensor)
        probabilities = torch.softmax(y_pred, dim=1)
        pred_classes = torch.argmax(y_pred, dim=1)

    # 打印结果
    for i, s in enumerate(input_strings):
        true_first_a = s.find('a') if 'a' in s else -1
        pred_class = pred_classes[i].item()

        position_desc = {
            0: "位置0 (第一个字符是'a')",
            1: "位置1 (第二个字符是'a')",
            2: "位置2 (第三个字符是'a')",
            3: "位置3 (第四个字符是'a')",
            4: "位置4 (第五个字符是'a')",
            5: "位置5 (第六个字符是'a')",
            6: "字符串中没有'a'"
        }

        prob_percent = probabilities[i][pred_class].item() * 100

        print(f"\n字符串: '{s}'")
        if true_first_a >= 0:
            print(f"  真实: 第一个'a'在位置 {true_first_a}")
        else:
            print(f"  真实: 字符串中没有'a'")

        print(f"  预测: {position_desc[pred_class]} (置信度: {prob_percent:.1f}%)")

        # 打印所有位置的概率
        if true_first_a >= 0 or pred_class != 6:
            print("  各个位置的预测概率:")
            for pos in range(sentence_length + 1):
                prob = probabilities[i][pos].item() * 100
                pos_desc = f"位置{pos}" if pos < sentence_length else "没有'a'"
                print(f"    {pos_desc}: {prob:.1f}%")


if __name__ == "__main__":
    main()

    # 测试预测功能
    print("\n测试预测功能...")
    test_strings = [
        "aabbcc",  # 位置0
        "baaccc",  # 位置2
        "bbaaac",  # 位置2
        "bbccca",  # 位置5
        "defghi",  # 没有'a'
        "xayaza",  # 位置1
        "aaaxxx",  # 位置0
        "xyzabc"  # 位置3
    ]

    predict("position_predictor.pth", "vocab.json", test_strings)
