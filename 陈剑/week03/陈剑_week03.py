import numpy as np
import torch
from torch import nn

#构造字符表
vocab = {
    "[pad]": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "p": 16,
    "q": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26,
    "[unk]": 27
}


class TorchRNN(nn.Module):
    def __init__(self, hidden_size, char_dim, sentence_length, vocab):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim, padding_idx=0)
        self.rnn = nn.RNN(char_dim, hidden_size, bias=False, batch_first=True)
        self.classifier = nn.Linear(hidden_size, sentence_length)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y_true=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        mean_pooled = torch.mean(rnn_out, dim=1)
        logits  = self.classifier(mean_pooled)
        if y_true is not None:
            return self.loss(logits, y_true)
        else:
            return logits


def build_sample(sentence_length):
    x = np.random.randint(1, len(vocab), np.random.randint(max(0, sentence_length - 8), sentence_length))
    if vocab["a"] not in x:
        x = np.append(x, vocab["a"])

    pad_length = sentence_length - len(x)
    x_padded = x
    if pad_length > 0:
        x_padded = np.pad(x, (0, pad_length), mode='constant', constant_values=0)
    return x_padded, np.where(x == vocab["a"])[0][0]


def build_dataset(sample_length, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(np.array(dataset_x)), torch.LongTensor(np.array(dataset_y))


def test(model, sentence_length, test_sample_size):
    model.eval()
    with torch.no_grad():
        correct_count = 0
        test_dataset_X, test_dataset_Y = build_dataset(test_sample_size, sentence_length=sentence_length)
        print(test_dataset_X.shape)
        for x, y in zip(test_dataset_X, test_dataset_Y):
            pred = model(x.unsqueeze(0))
           # print(f"测试数据: {x.tolist()}, 预测值: {pred.tolist()}")
            a_index = np.argmax(pred)
            if np.abs(a_index - y) < 0.5:
                correct_count += 1
    print("正确预测个数：%d, 正确率：%f" % (correct_count, correct_count / test_sample_size))


def main():
    epoch_num = 100
    batch_size = 30
    train_sample = 500
    char_dim = 20
    sentence_length = 12
    learning_rate = 0.005
    # 建立模型
    model = TorchRNN(hidden_size=15, vocab=vocab, char_dim=char_dim, sentence_length=sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
        test(model, test_sample_size=100, sentence_length=sentence_length)  # 测试本轮模型结果


main()
