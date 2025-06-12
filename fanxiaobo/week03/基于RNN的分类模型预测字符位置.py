import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class TorchNet(nn.Module):
    def __init__(self, vocab, vector_idm):
        super(TorchNet, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_idm, padding_idx=0)
        self.layer = nn.RNN(input_size=20, hidden_size=20, batch_first=True)
        self.Linear = nn.Linear(vector_idm, 7)
        self.loss = F.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        _, h = self.layer(x)
        x = h.squeeze()
        y_pred = self.Linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = 'qwertyuiopasdfghjklzxcvbnm'
    vocab = {'pad': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    if set('a') & set(x):
        y = x.index('a') + 1
    else:
        y = 0

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, sentence_length, vocab):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    x = np.asarray(X)
    y = np.asarray(Y)
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def test(model, sample_length, sentence_length, vocab):
    model.eval()
    x_test, y_test = build_dataset(sample_length, sentence_length, vocab)
    unique, count = np.unique(y_test, return_counts=True)
    converted_dict = dict(zip(unique, count))
    converted_dict = {int(w + 1): int(v) for w, v in converted_dict.items()}
    print(f'本次预测集中个样本分布-->{converted_dict}')
    correct, worry = 0, 0
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred = y_pred.argmax(dim=-1)
        for y_p, y_t in zip(y_pred, y_test):
            if y_p == int(y_t):
                correct += 1
            else:
                worry += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (worry + correct)))
    return correct / (worry + correct)


def main():
    vector_idm = 20
    learning_rate = 0.005
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    sentence_length = 6
    sample_length = 200

    vocab = build_vocab()
    model = TorchNet(vocab, vector_idm)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        loss_watch = []
        for batch_num in range(train_sample // batch_size):
            batch_x, batch_y = build_dataset(batch_size, sentence_length, vocab)
            loss = model(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_watch.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch, np.mean(loss_watch)))
        acc = test(model, sample_length, sentence_length, vocab)
        log.append([acc, np.mean(loss_watch)])

    torch.save(model.state_dict(), 'model.pth')
    # plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    # plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    # plt.legend()
    # plt.show()

    writer = open('vocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    return


def predict(model, vocab, input_strings):
    vector_idm = 20
    vocab = json.load(open('vocab.json', 'r', encoding='utf-8'))
    model = TorchNet(vocab, vector_idm)
    model.load_state_dict(torch.load('model.pth', weights_only=False))
    x = []
    for string in input_strings:
        x.append([vocab[char] for char in string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))

    for vec, res in zip(input_strings, result):
        print("输入：%s, 预测类别：%s" % (vec, np.argmax(res)))




if __name__ == '__main__':
    main()
    test_strings = ["dddasc", "wzadfg", "aqwdig", "nfkwww"]
    predict("model.pth", "vocab.json", test_strings)

