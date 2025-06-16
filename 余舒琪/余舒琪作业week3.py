import torch
import torch.nn as nn
import numpy as np
import random
import json

'''
基于pytorch的网络编写
判断文本中字符"a"出现的位置，其出现在第几个位置则为第几类样本
'''

class TorchModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 6)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        h, x = self.rnn(x)
        y_pred = self.linear(x.squeeze())
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 生成字符集
def build_vocab():
    chars = "a你我他sdfghjklzxcvbnmqwertyuiop"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab

# 随机生成一个文本长度为sentence_length且含字符"a"的样本，根据"a"出现的位置确定y
def build_sample(vocab, sentence_length):
    other_keys = [k for k in vocab.keys() if k != "a"]
    x = [random.choice(other_keys) for i in range(sentence_length - 1)]
    x.extend(["a"])
    random.shuffle(x)
    y = np.zeros(sentence_length)
    y[x.index("a")] = 1
    x = [vocab.get(word, vocab["unk"]) for word in x]
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

def build_model(vocab, char_dim, hidden_size):
    model = TorchModel(char_dim, hidden_size, vocab)
    return model

# 测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num, vocab, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数：{correct}，正确率：{correct / test_sample_num}")
    return correct / test_sample_num

def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 6
    hidden_size = 6
    sentence_length = 6
    learning_rate = 0.005
    vocab = build_vocab()
    model = build_model(vocab, char_dim, hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(train_sample, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"============\n第{epoch+1}轮平均loss：{np.mean(watch_loss)}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    print(log)
    # 保存模型
    torch.save(model.state_dict(), "mymodel2.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 6
    hidden_size = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    model = build_model(vocab, char_dim, hidden_size)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}，预测类别：{torch.argmax(result[i])}，输出：{result[i]}")

if __name__ == "__main__":
    main()
    test_strings = ["asdfgh", "你fhaij", "我ijkaw", "pakoih", "qwaros", "ouihja"]
    predict("mymodel2.pth", "vocab.json", test_strings)