"""

构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

"""
import json
import random

import torch
from torch import nn


# 1. 定义模型
class TorchModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, sentence_len):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, sentence_len)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, h = self.layer(x)
        output = output[:, -1, :]
        y_pred = self.linear(output)
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred


def useModel(sentence_len):
    # 每个字要转的维度
    char_dim = 10
    # RNN输出维度
    hidden_size = 20
    return TorchModel(sentence_len, char_dim, hidden_size, sentence_len)


# 2. 构建词表
def build_vocab():
    demo = "abc你好吗"
    vocab = {}
    for i in range(0, len(demo)):
        vocab[demo[i]] = i
    return vocab


# 3. 生成随机样本，作为训练数据
def build_example(vocab):
    # 随机生成一个包含a的sentence_len长度的句子，sentence_len要等于vocab的长度了
    sentence_len = len(vocab)
    random_sentence = random.sample(list(vocab.keys()), sentence_len)
    # 判断a所在的位置
    y = random_sentence.index("a")
    # 将每个字转换为向量
    x = [vocab.get(word) for word in random_sentence]
    print(f"生成的句子是{random_sentence}，转换成向量={x}, 所属类别={y}")
    return x, y


def build_dataset(vocab, data_num):
    data_x = []
    data_y = []
    for _ in range(data_num):
        x, y = build_example(vocab)
        data_x.append(x)
        data_y.append(y)

    return torch.LongTensor(data_x), torch.LongTensor(data_y)


# 4. 模型评价
def evaluate(model, vocab, eval_num):
    data_x, data_y = build_dataset(vocab, eval_num)
    model.eval()
    correct_num = 0
    wrong_num = 0
    with torch.no_grad():
        y_pred = model.forward(data_x)
        y_pred = torch.LongTensor([torch.argmax(y, dim=-1) for y in y_pred])
        print(f'数据{data_x}，预测结果{y_pred}')
        for y_p, y_t in zip(y_pred, data_y):
            if y_p == y_t:
                correct_num += 1
            else:
                wrong_num += 1

    return correct_num / (correct_num + wrong_num)


# 5. 训练模型
def main():
    # 定义参数
    # 轮次
    epoch_num = 5
    # 每批次数据量
    batch_num = 200
    # 总数据量
    data_num = 1000
    # 每个字要转的维度
    char_dim = 10
    # RNN输出维度
    hidden_size = 20
    # 构建词表
    vocab = build_vocab()
    # 生成训练数据集
    data_x, data_y = build_dataset(vocab, data_num)
    print(data_x, data_y)
    # 定义模型
    model = TorchModel(len(vocab), char_dim, hidden_size, len(vocab))
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 开始训练
    for epoch in range(epoch_num):
        model.train()

        for batch in range(data_num // batch_num):
            train_x = data_x[batch * batch_num: (batch + 1) * batch_num]
            train_y = data_y[batch * batch_num: (batch + 1) * batch_num]
            # 计算损失值
            loss = model.forward(train_x, train_y)
            # 计算梯度
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 梯度归零
            optimizer.zero_grad()

            # 查看损失值
            print(f"第{epoch}轮，第{batch}批，损失值loss={loss}")

    # 预测评价
    eval_num = 5
    correct_rate = evaluate(model, vocab, eval_num)
    print(f"预测{eval_num}条数据，正确率={correct_rate}")

    # 保存词表
    writer = open('myRNNWorkVocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False))
    writer.close()

    # 保存模型
    torch.save(model.state_dict(), 'myRNNWorkModel.pth')

    return


# 6. 使用模型预测
def predict(model_path, vocab_path, eval_sentences):
    # 读取词表
    reader = open(vocab_path, 'r', encoding='utf-8')
    vocab = json.loads(reader.read())
    reader.close()
    # 句子转为向量
    data_x = []
    for eval_sentence in eval_sentences:
        x = [vocab.get(word) for word in eval_sentence]
        data_x.append(x)

    # 使用模型
    sentence_len = len(vocab)
    model = useModel(sentence_len)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(torch.LongTensor(data_x))
        y_pred = torch.argmax(y_pred, dim=-1)
        for y_p, x in zip(y_pred, data_x):
            x_idx = x.index(0)
            print(f"数据{x}，预测结果={y_p}, 预测{x_idx == y_p}")



if __name__ == '__main__':
    # main()

    sentences = ["abc你好吗", "bc你好吗a", "bc你a好吗"]
    predict("myRNNWorkModel.pth", "myRNNWorkVocab.json", sentences)
