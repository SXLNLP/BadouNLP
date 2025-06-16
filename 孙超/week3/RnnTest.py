import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as fu
import torch.optim as optim
import numpy as np


class TorchModel(nn.Module):
    def __init__(self, vocab, vector_idm):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_idm, padding_idx=0) #embedding层
        self.layer = nn.RNN(input_size=20, hidden_size=20, batch_first=True) #RNN
        self.classify = nn.Linear(vector_idm, 6) #线性层 输出6类 0为没有
        self.loss = fu.cross_entropy
    def forward(self, x, y=None):
        x = self.embedding(x)
        _, h = self.layer(x)
        x = h.squeeze()
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)  #预测值和真实值计算损失
        else:
            return y_pred                #输出预测结果


def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
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
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    x = np.asarray(dataset_x)
    y = np.asarray(dataset_y)
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def evaluate(model, sample_length, sentence_length, vocab):
    model.eval()
    x_test, y_test = build_dataset(sample_length, sentence_length, vocab)
    unique, count = np.unique(y_test, return_counts=True)
    converted_dict = dict(zip(unique, count))
    converted_dict = {int(w + 1): int(v) for w, v in converted_dict.items()}
    print(f'本次预测集中个样本分布-->{converted_dict}')
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred = y_pred.argmax(dim=-1)
        for y_p, y_t in zip(y_pred, y_test):
            if y_p == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():

    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 5   #样本文本长度
    learning_rate = 0.005 #学习率
    sample_length = 200

    vocab = build_vocab()
    model = TorchModel(vocab, char_dim)
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
        acc = evaluate(model, sample_length, sentence_length, vocab)
        log.append([acc, np.mean(loss_watch)])

    torch.save(model.state_dict(), 'model.pth')

    writer = open('vocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    return


def predict(model, vocab, input_strings):
    vector_idm = 20
    vocab = json.load(open('vocab.json', 'r', encoding='utf-8'))
    model = TorchModel(vocab, vector_idm)
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
    test_strings = ["ewuoa", "wurya", "awqse", "gooua","wwaww"]
    predict("model.pth", "vocab.json", test_strings)
