import torch
import torch.nn as nn
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN层
        self.classify = nn.Linear(vector_dim, 6)  # 六分类输出
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim)

        # 取序列中每个时间步的输出
        # 我们关注每个位置是否首次出现'a'，所以使用所有时间步的输出
        y_pred = self.classify(rnn_out)  # (batch_size, sen_len, 6)

        # 只取序列中每个位置对应的预测
        if y is not None:
            # 将y从(batch_size, sen_len)转为(batch_size * sen_len)
            # 同时将预测reshape为(batch_size * sen_len, 6)
            return self.loss(y_pred.view(-1, 6), y.view(-1))
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # 随机生成6字符的字符串
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 创建标签：记录每个位置是否是'a'首次出现的位置
    # 0表示不是首次出现位置，1-5表示在位置1-5首次出现
    y = [0] * sentence_length
    found_first = False

    for i, char in enumerate(x):
        if char == 'a' and not found_first:
            y[i] = i + 1  # 位置索引从1开始
            found_first = True

    # 如果没有找到'a'，所有位置标签保持0
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim):
    model = TorchModel(char_dim, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)

    # 统计真实标签中的类别分布
    y_flat = y.view(-1)
    class_counts = [(y_flat == i).sum().item() for i in range(6)]
    print("类别分布（0-5）:", class_counts)

    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        # 计算每个位置是否正确预测
        for i in range(y.shape[0]):  # batch维度
            for j in range(y.shape[1]):  # 序列长度维度
                pred_class = torch.argmax(y_pred[i, j]).item()
                true_class = y[i, j].item()
                if pred_class == true_class:
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"正确预测位置数：{correct}/{total}, 准确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 2000  # 每轮训练总共训练的样本总数
    char_dim = 64  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            y = y.clamp(0, 5)  # 确保标签在0-5范围内
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / (train_sample / batch_size)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        evaluate(model, vocab, sentence_length)

    torch.save(model.state_dict(), "rnn_model.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("训练完成，模型已保存")


def predict(model_path, vocab_path, input_strings):
    char_dim = 64
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 处理输入
    x = []
    for s in input_strings:
        # 截断或填充到6个字符
        s = s[:sentence_length].ljust(sentence_length, 'x')[:sentence_length]
        x.append([vocab.get(char, vocab['unk']) for char in s])

    with torch.no_grad():
        y_pred = model(torch.LongTensor(x))

    # 解析预测结果
    for i, s in enumerate(input_strings):
        print(f"\n输入: '{s}'")
        for pos, char in enumerate(s):
            if pos >= sentence_length:
                break
            probs = torch.softmax(y_pred[i, pos], dim=0)
            pred_class = torch.argmax(probs).item()
            if pred_class == 0:
                print(f"  位置 {pos + 1}: '{char}' - 非首次'a'位置")
            else:
                print(f"  位置 {pos + 1}: '{char}' - 预测为首次'a'出现位置 (类别 {pred_class})")


if __name__ == "__main__":
    main()

    # 测试字符串
    test_strings = [
        "bcdefg",  # 无a
        "axxxxx",  # a在位置1
        "xaxxxx",  # a在位置2
        "xxaxxx",  # a在位置3
        "xxxaxx",  # a在位置4
        "xxxxax",  # a在位置5
        "xxxxxa",  # a在位置6
        "aaxxxx",  # 多个a，首次在位置1
        "xaaaxx"  # 多个a，首次在位置2
    ]

    predict("rnn_model.pth", "vocab.json", test_strings)
