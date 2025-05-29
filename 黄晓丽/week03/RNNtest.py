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


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)  # RNN层
        self.classify = nn.Linear(hidden_size, sentence_length + 1)  # 线性层
        self.sentence_length = sentence_length
        self.loss = nn.CrossEntropyLoss()  # 多分类使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)  # RNN处理序列
        last_output = output[:, -1, :]  # 取最后一个时间步的输出
        y_pred = self.classify(last_output)  # 分类预测
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他defaghijklmnopqrstuvwxyz!@#$%^&*"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 生成不包含'a'的字符串
    non_a_chars = [c for c in vocab.keys() if c != 'a' and c not in ['pad', 'unk']]
    x = [random.choice(non_a_chars) for _ in range(sentence_length)]

    # 随机选择一个位置插入'a'
    if random.random() > 0.2:  # 80%的概率包含'a'
        pos = random.randint(0, sentence_length - 1)
        x[pos] = 'a'
    else:  # 20%的概率不包含'a'
        pos = sentence_length  # 没有出现，位置为字符串长度

    # 将字符转换为索引
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, pos


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
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


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    y = y.squeeze()
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predictions = torch.argmax(y_pred, dim=1)  # 获取预测类别
        for y_pred, true in zip(predictions, y):  # 与真实标签进行对比
            if y_pred == true:
                correct += 1
            else:
                wrong += 1
    acuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, acuracy))
    return acuracy


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 3000  # 每轮训练总共训练的样本总数
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
    return log


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 32  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    padding_char = '我' if '我' in vocab else list(vocab.keys())[1]
    for input_string in input_strings:
        # 截断或填充字符串
        input_string = input_string[:sentence_length].ljust(sentence_length, padding_char)
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.LongTensor(x))
        predictions = torch.argmax(result, dim=1)

    for i, input_string in enumerate(input_strings):
        pred_pos = predictions[i].item()
        if pred_pos == sentence_length: # 超过六个字符就当未出现
            print(f"字符串: '{input_string}' -> 'a'未出现")
        else:
            print(f"字符串: '{input_string}' -> 'a'首次出现在位置: {pred_pos}")

if __name__ == "__main__":
    # main()
    test_strings = ["fnavfeae", "wz你dfsag", "arqwdesg", "n我kwsww"]
    predict("model.pth", "vocab.json", test_strings)
