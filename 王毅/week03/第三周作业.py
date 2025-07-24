import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use('TkAgg')  # 或使用 'Qt5Agg'
import random

"""
要求：
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # Embedding层
        x = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, vector_dim)
        # RNN层
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_size)
        # 取最后一个时间步的输出
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden_size)
        # 分类层
        logits = self.classifier(last_hidden)  # (batch, num_classes)
        if y is not None:
            # 训练模式：计算损失
            return self.loss(logits, torch.argmax(y, dim=1))
        else:
            # 预测模式：返回softmax概率
            return nn.functional.softmax(logits, dim=1)


# 定义词表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index  # 每个字母对应一个序号
    return vocab

#随机生成一个样本
def build_sample(vocab, num):
    # 获取所有字符键，排除'a'
    chars = [k for k in vocab.keys() if k != 'a']
    # 随机选择num-1个字符
    selected = random.sample(chars, num - 1)
    # 加入'a'构成Q
    x = selected + ['a']
    # 打乱顺序
    random.shuffle(x)
    # 找到'a'的位置
    a_index = x.index('a')
    # 生成one-hot向量
    y = [0] * num
    y[a_index] = 1
    return x, y

#建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        # 将字符转换为索引
        dataset_x.append([vocab[char] for char in x])
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model( char_dim, sentence_length,hidden_size,vocab):
    model = TorchModel(
        vector_dim=char_dim,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_classes=sentence_length  # 类别数等于序列长度
    )
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本

    # 找出每行中值最大的索引位置-----真实值
    class_indices = torch.argmax(y, dim=1)
    print("class_indices:~~~~~~~~~~")
    print(class_indices)

    # 统计真实的y每一类的数量
    class_counts = torch.bincount(class_indices, minlength=5)
    print("打印真实的y，每一类的统计数量：")
    for i in range(6):
        print(f"类别 {i}: {int(class_counts[i])}个")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if np.argmax(y_p) == np.argmax(y_t):  # 如果两个五维向量是相等的，则认为，预测值准确
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 10  # 每个字的维度
    hidden_size=10
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(char_dim, sentence_length,hidden_size,vocab)
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
    # 加载词汇表
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # 初始化模型（需补充完整参数）
    model = TorchModel(vector_dim=10, hidden_size=10,
                       vocab_size=len(vocab), num_classes=6)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        for chars in input_strings:
            input_indices = torch.LongTensor([[vocab[c] for c in chars]])
            probs = model(input_indices)
            pred_class = torch.argmax(probs).item()
            print(f"输入: {chars}, 预测位置: {pred_class}")


if __name__ == "__main__":
    main()
    # test_strings = ["anvfee", "wasdfg", "rqwaeg", "nskwwa"]
    # predict( "model.pth", "vocab.json", test_strings)
