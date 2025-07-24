import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
韦杏仪 第三周 构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        # self.classify = nn.Linear(vector_dim, sentence_length)  # 线性层，输出维度等于句子长度，每一个位置代表一个类别
        # self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失
        self.rnn = nn.RNN(vector_dim, 64, batch_first=True)  # RNN层
        self.classify = nn.Linear(64, sentence_length)  # 线性层，输出每个位置的概率
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        # x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, 1) 3*20 20*1 -> 3*1
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        output, _ = self.rnn(x)
        x = self.classify(output[:, -1, :])
        if y is not None:
            # return self.loss(y_pred, y)  # 预测值和真实值计算损失
            return self.loss(x, y)
        else:
            # return y_pred  # 输出预测结果
            return torch.softmax(x, dim=1)  # 输出各位置概率


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = generate_random_charset(30)  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 获取所有实际字符（排除特殊标记）
    actual_chars = [char for char in vocab.keys() if char not in ["pad", "unk"]]
    # 确保至少有一个"a"
    has_a = False
    while not has_a:
        # 随机从实际字符中选取sentence_length个字，可能重复
        x = [random.choice(actual_chars) for _ in range(sentence_length)]
        has_a = 'a' in x
    # 确定"a"第一次出现的位置，确保索引在0到sentence_length-1之间
    y = next((i for i, c in enumerate(x) if c == 'a'), sentence_length - 1)

    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        # 将字符转换为对应的索引
        x_indices = [vocab.get(word, vocab['unk']) for word in x]
        dataset_x.append(x_indices)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    # with torch.no_grad():
    #     y_pred = model(x)  # 模型预测
    #     for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
    #         if float(y_p) < 0.5 and int(y_t) == 0:
    #             correct += 1  # 负样本判断正确
    #         elif float(y_p) >= 0.5 and int(y_t) == 1:
    #             correct += 1  # 正样本判断正确
    #         else:
    #             wrong += 1
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 获取概率最大的位置

        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    # print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 随机生成包含字母 'a' 的字符串
def generate_random_charset(charset_size=30):
    chars = ['a']
    while len(chars) < charset_size:
        char = chr(random.randint(33, 126))
        if char not in chars:
            chars.append(char)
    return ''.join(chars)

def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()

    # 打印生成的字符集
    chars = [char for char, idx in vocab.items() if idx not in [0, len(vocab) - 1]]
    # print(f"生成的字符集: {''.join(chars)}")

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
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        # 确保输入字符串长度与模型训练时一致
        if len(input_string) < sentence_length:
            input_string = input_string + ' ' * (sentence_length - len(input_string))
        elif len(input_string) > sentence_length:
            input_string = input_string[:sentence_length]

        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        result = torch.argmax(result, dim=1)  # 获取概率最大的位置

    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测'a'的位置：%d" % (input_string, result[i]))


if __name__ == "__main__":
    main()
    test_strings = ["abbb", "bbbbabbb", "bbbaaaaaa", "baaaaaaa"]
    predict("model.pth", "vocab.json", test_strings)
