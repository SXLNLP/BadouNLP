import torch
import torch.nn as nn
import numpy as np
import random
import json


class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, vector_dim)
        _, h_n = self.rnn(x)  # h_n: (1, batch_size, vector_dim)
        h_n = h_n.squeeze(0)  # (batch_size, vector_dim)
        y_pred = self.classify(h_n)  # (batch_size, num_classes)

        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 返回概率分布

#知识库
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

#构造单一测试用例
def build_sample(vocab, sentence_length):
    # 随机生成字符串，确保包含至少一个'a'
    while True:
        x = [random.choice(list(vocab.keys())[1:-1]) for _ in range(sentence_length)]
        if 'a' in x:
            break

    # 找到第一个'a'的位置（从0开始）
    first_a_pos = x.index('a')
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, first_a_pos

#构造测试集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
#建立模型
def build_model(vocab, char_dim, sentence_length):
    num_classes = sentence_length  # 类别数为字符串长度（位置0到sentence_length-1）
    model = RNNModel(char_dim, sentence_length, len(vocab), num_classes)
    return model

#测试方法测试用
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print(f"样本分布: {torch.bincount(y).tolist()}")

    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y).sum().item()

    accuracy = correct / len(y)
    print(f"正确预测个数: {correct}, 正确率: {accuracy:.4f}")
    return accuracy

#训练方法
def main():
    # 配置参数
    epoch_num = 20
    batch_size = 32
    train_sample = 2000
    char_dim = 64
    sentence_length = 10  # 字符串长度
    learning_rate = 0.001

    # 建立知识库
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

        avg_loss = np.mean(watch_loss)
        print(f"第{epoch + 1}轮平均loss: {avg_loss:.4f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, avg_loss])

    # 保存模型和词表
    torch.save(model.state_dict(), "rnn_model.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

#预测方法
def predict(model_path, vocab_path, input_strings):
    char_dim = 64
    sentence_length = 10
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    # 预处理输入
    x = []
    for s in input_strings:
        s = s[:sentence_length].ljust(sentence_length, ' ')  # 截断或填充
        x.append([vocab.get(c, vocab['unk']) for c in s])
    x = torch.LongTensor(x)

    model.eval()
    with torch.no_grad():
        probs = model(x)
        predictions = torch.argmax(probs, dim=1)

    for i, s in enumerate(input_strings):
        print(f"输入: '{s}', 预测第一个'a'出现的位置在第 : {predictions[i].item()+1} 位")


if __name__ == "__main__":
    main()
    test_strings = ["adsdsd", "basdsadasda", "scadfafa", "dsdadsfsv"]
    predict("rnn_model.pth", "vocab.json", test_strings)
