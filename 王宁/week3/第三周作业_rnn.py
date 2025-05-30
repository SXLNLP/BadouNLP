# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json



class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层 字符串转为矩阵
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # 线性层 制定字符未出现时输出为字符串长度
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentence_length) -> (batch_size, sentence_length, vector_dim)
        output, hn = self.rnn(x) # (batch_size, sen_len, vector_dim)
        last_output = output[:, -1, :] # 最后一个时间步的隐藏状态 (batch_size, vector_dim)
        logits = self.classify(last_output)  # (batch_size, sentence_length + 1)
        if y is not None:
            return self.loss(logits, y.squeeze())  # 预测值和真实值计算损失
        else:
            return torch.softmax(logits, dim=1)  # 输出预测结果


# 为每个字生成一个标号
# abc -> [1,2,3]
def build_vocab():
    chars = "富强民主文明和谐自由平等公正法治爱国敬业诚信友善"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 随机决定是否插入目标字符
    if random.random() < 0.95:
        # 随机选择插入位置（0到sentence_length-1）
        insert_index = random.randint(0, sentence_length - 1)
        x[insert_index] = '爱'

    # 获取指定字符出现的下标
    try:
        y = x.index("爱")  # 返回第一次出现下标
    except ValueError:
        y = sentence_length  # 指定字未出现，返回文本长度

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本

    # 统计各类别样本数
    class_count = [0] * (sample_length + 1)
    for label in y:
        class_count[label.item()] += 1
    print("各类别样本数量:", class_count)

    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y.squeeze()).sum().item()

    accuracy = correct / y.size(0)
    print(f"正确预测个数: {correct}, 正确率: {accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        evaluate(model, vocab, sentence_length)  # 测试本轮模型结果


    # 保存模型
    torch.save(model.state_dict(), "rnn_model_03.pth")
    # 保存词表
    writer = open("rnn_vocab.json", "w", encoding="utf8")
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
    for s in input_strings:
        s = s[:sentence_length].ljust(sentence_length, 'x')
        encoded = [vocab.get(char, vocab['unk']) for char in s]
        x.append(encoded)  # 将输入序列化

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度

        result = model.forward(torch.LongTensor(x))  # 模型预测
        pre_index = torch.argmax(result, dim=1)

    for i, input_string in enumerate(input_strings):
        msg = "未出现" if pre_index[i] >= sentence_length else " 类"
        print(f"输入：'{input_string}', 预测类别：'{pre_index[i]} {msg}',  概率值：'{result[i]}'" )  # 打印结果


if __name__ == "__main__":
    # main()
    test_strings = [
        "爱国公正平等",
        "a爱",
        "ab爱",
        "友善a爱",
        "自由和谐爱国",
        "abcde爱",
        "富强民主文明",
        "文明诚信敬业df"]
    predict("rnn_model_03.pth", "rnn_vocab.json", test_strings)
    # print(build_vocab())
