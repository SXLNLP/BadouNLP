import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
"""


class CharPositionModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(CharPositionModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化层
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # 线性层，输出类别数为句子长度+1
        self.loss = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss代替手动softmax

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=-1)  # 输出预测概率分布


def build_vocab():
    chars = "abcdefghijk"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = random.choices(list(vocab.keys())[1:-1], k=sentence_length)  # 随机采样，排除pad和unk
    y = x.index("a") if "a" in x else sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = CharPositionModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y).sum().item()
    accuracy = correct / len(y)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 增加训练轮数
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss:{avg_loss:.4f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, avg_loss])

    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    # 处理输入字符串，确保长度为sentence_length
    processed_inputs = []
    for s in input_strings:
        if len(s) < sentence_length:
            s += 'pad' * (sentence_length - len(s))  # 填充不足部分
        s = s[:sentence_length]  # 截断超长部分
        processed_inputs.append([vocab.get(c, vocab['unk']) for c in s])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(processed_inputs))

    for i, input_str in enumerate(input_strings):
        pred_class = torch.argmax(result[i]).item()
        prob = torch.max(result[i]).item()
        if pred_class == sentence_length:
            print(f"输入：{input_str}, 预测结果：字符串中没有'a'")
        else:
            print(f"输入：{input_str}, 预测结果：'a'在第{pred_class}个位置, 概率值：{prob:.4f}")


if __name__ == "__main__":
    main()
    # test_strings = ["abcdef", "bcadeg", "ghijka", "bbbbbb", "a", "xyz", "padpad"]
    # predict("model.pth", "vocab.json", test_strings)
