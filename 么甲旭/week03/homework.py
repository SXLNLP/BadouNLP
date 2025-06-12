# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断字符 'a' 第一次出现在字符串中的位置

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, 64, batch_first=True)
        self.classifier = nn.Linear(64, sentence_length + 1)  # 分类器输出维度为句子长度+1(包含0类)
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数处理多分类问题

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)  # 获取RNN输出 (batch_size, sen_len, hidden_dim)
        output = output[:, -1, :]  # 取最后一个时间步的输出 (batch_size, hidden_dim)
        y_pred = self.classifier(output)  # 通过分类器得到预测分数 (batch_size, sentence_length+1)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出各类别的概率分布


# 字符集添加字符 'a'
def build_vocab():
    chars = "a你我他defghijklmnopqrstuvwxyz"  # 添加字符 'a' 到字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 27
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 有一定概率包含 'a'，并记录其首次出现的位置作为类别
def build_sample(vocab, sentence_length):
    # 80%的概率包含 'a'
    if random.random() < 0.8:
        # 确保至少有一个 'a'
        # 修正：先将vocab.keys()转换为集合，再进行减法操作
        valid_chars = set(vocab.keys()) - {'pad', 'unk', 'a'}
        x = [random.choice(list(valid_chars)) for _ in range(sentence_length - 1)]
        a_position = random.randint(0, sentence_length - 1)  # 随机选择 'a' 出现的位置
        x.insert(a_position, 'a')  # 在随机位置插入 'a'
        y = a_position + 1  # 类别为 'a' 第一次出现的位置索引+1
    else:
        # 不包含 'a' 的样本
        # 修正：先将vocab.keys()转换为集合，再进行减法操作
        valid_chars = set(vocab.keys()) - {'pad', 'unk', 'a'}
        x = [random.choice(list(valid_chars)) for _ in range(sentence_length)]
        y = 0  # 类别为0表示没有 'a'

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 多分类任务中标签不需要包装为列表
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, len(vocab))
    return model


# 测试代码
def evaluate(model, vocab, sample_length, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本

    # 统计各类别的样本数
    class_counts = {i: 0 for i in range(sentence_length + 1)}
    for label in y:
        class_counts[label.item()] += 1

    print("测试集类别分布:")
    for cls, count in class_counts.items():
        print(f"类别 {cls}: {count} 个样本")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，输出概率分布
        y_pred_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别

        for y_p, y_t in zip(y_pred_classes, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong):.6f}")
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 增加训练轮数
    batch_size = 20
    train_sample = 1000  # 增加训练样本数
    char_dim = 20
    sentence_length = 5  # 样本文本长度
    learning_rate = 0.001  # 降低学习率

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

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.6f}")
        acc = evaluate(model, vocab, train_sample, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 5
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    x = []
    for input_string in input_strings:
        # 处理输入字符串
        processed = []
        for char in input_string:
            if char in vocab:
                processed.append(vocab[char])
            else:
                processed.append(vocab['unk'])

        # 填充或截断到固定长度
        if len(processed) < sentence_length:
            processed += [vocab['pad']] * (sentence_length - len(processed))
        else:
            processed = processed[:sentence_length]

        x.append(processed)

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))  # 获取概率分布

    for i, input_string in enumerate(input_strings):
        predicted_class = torch.argmax(result[i]).item()  # 获取预测的类别
        confidence = result[i][predicted_class].item()  # 获取对应类别的置信度

        # 解释预测结果
        if predicted_class == 0:
            explanation = "未检测到字符 'a'"
        else:
            explanation = f"字符 'a' 首次出现在位置 {predicted_class - 1}"

        print(f"输入：{input_string}, 预测类别：{predicted_class}, 置信度：{confidence:.6f}, 解释：{explanation}")


if __name__ == "__main__":
    main()
    # 测试预测功能
    test_strings = ["a你好世界", "你a好世界", "你好a世界", "你好世a界", "你好世界a", "你好世界", "aaaaa"]
    predict("model.pth", "vocab.json", test_strings)
