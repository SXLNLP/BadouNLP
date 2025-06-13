
import torch
import torch.nn as nn
import numpy as np
import random
import json
from sklearn.metrics import accuracy_score


class TorchModel(nn.Module):
    def __init__(self, vector_dim, max_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # padding_idx=0表示用0填充
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, num_classes)  # 输出21类(0-20)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, length) -> (batch_size, length, vector_dim)
        output, _ = self.rnn(x)
        # 取最后一个时间步的输出作为分类依据
        x = output[:, -1, :]  # (batch_size, vector_dim)
        y_pred = self.classify(x)  # (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())
        else:
            return torch.softmax(y_pred, dim=1)



def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, max_length):
    # 控制5%的样本没有'a'，95%的样本包含'a'
    has_a = random.random() > 0.05

    if has_a:
        # 先生成不含a的部分
        prefix_len = random.randint(0, max_length-1)
        x = [random.choice(list(set(vocab.keys())-{'a','pad'})) for _ in range(prefix_len)]
        # 插入一个a
        x.append('a')
        # 补充剩余部分
        suffix_len = random.randint(0, max_length - len(x))
        x += [random.choice(list(set(vocab.keys()))) for _ in range(suffix_len)]
        y = prefix_len + 1  # a的位置
    else:
        # 生成不包含'a'的样本（占5%）
        length = random.randint(1, max_length * 2)
        x = [random.choice(list(set(vocab.keys()) - {'a', 'pad'})) for _ in range(length)]
        y = 0  # 不存在

    # 处理长度不足或超长
    if len(x) < max_length:
        x = x + [vocab['pad']] * (max_length - len(x))  # 补0
    else:
        x = x[:max_length]  # 截断

    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y


def build_dataset(sample_length, vocab, max_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, max_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def evaluate(model, vocab, max_length):
    model.eval()
    x, y_true = build_dataset(200, vocab, max_length)

    with torch.no_grad():
        y_pred = model(x)
        y_pred_classes = torch.argmax(y_pred, dim=1)

    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"各类别样本数量: {np.bincount(y_true.flatten())}")
    print(f"正确预测个数: {accuracy_score(y_true, y_pred_classes, normalize=False)}/{len(y_true)}")
    print(f"正确率: {accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 40
    batch_size = 32
    train_sample = 2000
    char_dim = 32
    max_length = 20  # 最大长度20
    num_classes = max_length + 1  # 0-20共21类
    learning_rate = 0.001

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(char_dim, max_length, vocab, num_classes)

    # 训练
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, max_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, max_length)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型和词汇表
    torch.save(model.state_dict(), "rnn_model_v2.pth")
    with open("rnn_vocab_v2.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def predict(model_path, vocab_path, input_strings, max_length=20):
    char_dim = 32
    num_classes = max_length + 1

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = TorchModel(char_dim, max_length, vocab, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 处理输入字符串
    processed = []
    for s in input_strings:
        s = s.lower()
        # 转换为ID序列
        seq = [vocab.get(c, vocab['unk']) for c in s]
        # 处理长度
        if len(seq) < max_length:
            seq = seq + [vocab['pad']] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        processed.append(seq)

    with torch.no_grad():
        x = torch.LongTensor(processed)
        probs = model(x)
        pred_classes = torch.argmax(probs, dim=1)

        for s, cls, prob in zip(input_strings, pred_classes, probs):
            pos = "不存在" if cls == 0 else f"第{cls}位"
            print(f"输入: '{s}', 预测第一个'a'位置: {pos}, 各类别概率: {prob.tolist()}")


if __name__ == "__main__":
    main()
    test_strings = [
        "banana",
        "apple",
        "orange",
        "1234567890",
        "a quick brown fox",
        "no a in this string",
        "a" * 25  # 测试超长字符串
    ]
    predict("rnn_model_v2.pth", "rnn_vocab_v2.json", test_strings)
