import torch
import torch.nn as nn
import numpy as np
import random
import json

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # Embedding层
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=128, num_layers=1, batch_first=True)  # RNN层
        self.classify = nn.Linear(128, num_classes)  # 线性层，输出多分类结果
        # self.activation = nn.Softmax(dim=1)  # Softmax激活函数
        self.loss = nn.CrossEntropyLoss()  # 损失函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        x = x[:, -1, :]  # 取RNN的最后一个时间步的输出
        x = self.classify(x)  # (batch_size, hidden_size) -> (batch_size, num_classes)
        # y_pred = self.activation(x)  # (batch_size, num_classes) -> (batch_size, num_classes)

        if y is not None:
            return self.loss(x, y.long())  # 确保y是LongTensor
        else:
            return torch.softmax(x, dim=1)  # 输出预测结果

def build_vocab():
    chars = "你我他adefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if "a" in x:
        y = x.index('a')
    else:
        y = sentence_length  # 如果没有"a"，则位置为句子长度
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TorchModel(char_dim, sentence_length, vocab, num_classes)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        y_pred_class = torch.argmax(y_pred, dim=1)
        for y_p, y_t in zip(y_pred_class, y):
            if int(y_t) == y_p.item():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 1000
    char_dim = 20
    sentence_length = 6
    num_classes = sentence_length + 1  # 类别数为句子长度 + 1
    learning_rate = 0.001

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

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
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return model, vocab

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length, sentence_length + 1)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        result_class = torch.argmax(result, dim=1)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, result_class[i].item(), result[i][result_class[i]].item()))

if __name__ == "__main__":
    main()
    test_strings = ["anvfee", "wz你afg", "rawdeg", "n我kwwa"]
    predict("model.pth", "vocab.json", test_strings)
