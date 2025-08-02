import json
import torch
import torch.nn as nn
import numpy as np
import random


"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # 1) Embedding 层
        # 输入形状: (batch, sentence_length)  dtype=int64
        # 输出形状: (batch, sentence_length, vector_dim)
        self.embedding = nn.Embedding(len(vocab), vector_dim)

        # 2.1) 全局平均池化（沿 seq_len 维）
        # 输入形状: (batch, vector_dim, seq_len)  <- 注意顺序
        # 输出形状: (batch, vector_dim, 1)
        self.pool = nn.AvgPool1d(sentence_length)

        # self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)

        # 3) 分类头
        # 输入形状: (batch, vector_dim)  <- squeeze 后
        # 输出形状: (batch, sentence_length+1)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)           # x: (batch, seq_len) -> (batch, seq_len, vector_dim)

        # rnn_out, hidden = self.rnn(x)
        # x = rnn_out[:, -1, :]


        x = x.transpose(1,2)            # (batch, vector_dim, seq_len)
        # print(x.shape)
        x = self.pool(x)                # (batch, vector_dim, 1)
        # print(x.shape)
        x = x.squeeze(-1)               # (batch, vector_dim)
        # print(x.shape)

        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 为每个字生成一个标号
# abc -> [1,2,3]
def build_vocab():
    chars = "富强民主文明和谐自由平等公正法治爱国敬业诚信友善abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

# 随机生成单个样本
# 从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 随机决定是否插入目标字符
    if random.random() < 0.95:
        # 随机选择插入位置（0到sentence_length-1）
        insert_index = random.randint(0, sentence_length - 1)
        x[insert_index] = '爱'

    # 获取指定字符出现的下标
    try:
        y = x.index("爱")  # 返回第一次出现下标
    except ValueError:
        y = sentence_length  # 指定字未出现，返回文本长度

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

#建立数据集
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
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(sample_length, vocab, sample_length)
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        # print("测试预测集：", x)
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    #参数配置
    epoch_num = 20
    batch_size = 40
    train_sample_length = 1000
    char_dim = 30
    sentence_length = 10
    learning_rate = 0.001

    #建立词表
    vocab = build_vocab()

    #建立模型
    model = build_model(vocab, char_dim, sentence_length)

    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample_length / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        evaluate(model, vocab, sentence_length)  # 测试本轮模型结果

    #保存模型
    torch.save(model.state_dict(), "model_review.pth")

    #保存词表
    writer = open("vocab_review.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    return

#使用模型预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30
    sentence_length = 10
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for s in input_strings:
        s = s[:sentence_length].ljust(sentence_length, 'x')
        encoded = [vocab.get(char, vocab['unk']) for char in s]
        x.append(encoded)  # 将输入序列化 并统一长度
    model.eval()
    with torch.no_grad():
        print(x)
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))

if __name__ == "__main__":
    main()

    test_strings = [
        "爱国公正平等",
        "a爱",
        "ab爱",
        "友善a爱",
        "自由和谐爱国",
        "abcde爱",
        "富强民主文明",
        "文明诚信敬业df"]
    predict("model_review.pth", "vocab_review.json", test_strings)
