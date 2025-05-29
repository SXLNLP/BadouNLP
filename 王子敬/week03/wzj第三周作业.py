import torch
import torch.nn as nn
import numpy as np
import random
import json


class TorchRNN(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # Embedding层
        # RNN中不再需要池化
        # self.pool = nn.AvgPool1d(sentence_length)  # 池化层

        # 由于多了一个不存在a的情况，似乎要对sentence_length+1，否则面对将a不存在和a在第一位时会出现都赋值为0的情况，
        # 同时如果多1维度，直接对下面的索引位置进行+1处理，在计算时loss会超

        # 与线性层不同，RNN返回的是一个元组（tuple）返回内容是output, h_n = self.layer(x)
        # 其中output: 所有时间步的输出，形状是 (batch_size, seq_len, hidden_size)
        # h_n: 最后一个时间步的隐藏状态，形状是 (num_layers * num_directions, batch, hidden_size)
        # self.layer = nn.RNN(vector_dim, 128, batch_first=True)
        self.layer = nn.RNN(vector_dim, 128, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(128, sentence_length+1)
        self.loss = nn.CrossEntropyLoss()
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, _ = self.layer(x)
        # output1, _ = self.layer(x)
        # output1 = self.dropout(output1)
        # output2, _ = self.layer(output1)
        y_pred = self.fc(output[:, -1, :])  # 取最后一个时间步
        # y_pred = self.activation(x)  # 使用了交叉熵损失
        if y is not None:
            return self.loss(y_pred, y)  # 使用预测值和真实值计算损失
        else:
            return y_pred


#  建立字符集，（实际中应对为数据集中的每一个字符生成一个对应序号。
#  还应该注意是否区分大小写，不同字符、编码等问题
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz赵钱孙李周吴郑王子鼠丑牛寅虎卯兔辰龙巳蛇午马未羊申猴酉鸡戌狗亥猪甲乙丙丁戊己庚辛壬癸"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    # print("字符集的大小为：", len(vocab))
    vocab['unk'] = len(vocab)
    return vocab


#  随机生成样本
def build_sample(vocab, sentence_length):
    # 随机从字符集中选sentence_length个字组成句子，字符允许重复（暂时对语序等不做要求）
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 当字符a不在列表中时设置为0，不知道这样的设置是否会影响模型的训练
    if 'a' not in x:
        y = 0
    else:
        y = x.index('a')+1  # 考虑使用异常处理来应对当元素值不在列表中时抛出的ValueError异常
    # 对这里x字符转序号的操作不知道是否应该对序号+1，因为前面0被不存在的情况占用了，对一一对应关系的影响尚不清楚
    x = [vocab.get(word, vocab['unk']) for word in x]  # 字符转序号，用于embedding层
    return x, y


# 建立数据集， 按需要的样本数量进行生成。
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchRNN(char_dim, sentence_length, vocab)
    return model


# 测试代码，测试本轮模型准确率
def evaluate(model, vocab, sample_length):
    model.eval()  # 测试模式
    # 测试样本数量为200, 这里的第二个sample_length其实是sentence_length
    x, y = build_dataset(200, vocab, sample_length)
    print("正在执行本轮模型测试")
    correct = 0  # 计算预测正确的个数
    with torch.no_grad():
        y_pred = model(x)  # 进行预测
        for y_p, y_t in zip(y_pred, y):
            # 由于上面对维度问题和位置问题进行过处理这里不需要在y_p.argmax()后面+1了
            if y_p.argmax() == y_t:  # 当值为0时似乎会有问题，需要修改上面build_sample中的逻辑
                correct += 1
    accuracy = correct / 200
    # print("correct:%d" % correct)
    print("测试完成，正确个数为：%d，正确率为：%f" % (correct, accuracy))
    return accuracy


def main():
    # 设置模型参数
    epoch_num = 10          # 训练轮数
    batch_size = 20         # 每次训练样本个数
    train_sample = 5000      # 每轮训练的总样本数
    char_dim = 20           # 每个字符的维度
    sentence_length = 20    # 样本文本长度
    learning_rate = 0.001   # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 记录训练过程
    log =[]
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 训练模式
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零

            watch_loss.append(loss.item())

        print('---------------------------------')
        print("第%d轮平均loss值为：%f" % (epoch + 1, np.mean(watch_loss)))
        accuracy = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([accuracy, np.mean(watch_loss)])
        print("正在打印log：", log)

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    # 保存词表
    writer = open("vocab.txt", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字符的维度
    sentence_length = 20  # 样本文本长度
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)       # 建立模型
    model.load_state_dict(torch.load(model_path))               # 加载训练好的模型
    x = []
    for input_string in input_strings:
        char_ids = [vocab.get(char, vocab['unk']) for char in input_string]  # 字符串转序号，unk用于代替不存在于vocab中的字符
        if len(char_ids) < sentence_length:
            char_ids += [vocab['pad']] * (sentence_length - len(char_ids))  # 当长度不足sentence_length时用pad补齐
        else:
            char_ids = char_ids[:sentence_length]   # 长度过长时截断
        x.append(char_ids)
        # x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        result = nn.Softmax(dim=-1)(result)  # 为了可读性进行一次Softmax
    for i, input_string in enumerate(input_strings):
        y_pred = torch.argmax(result[i]).item()
        confidence = result[i][y_pred].item()
        print("输入的字符串为：%s，预测类别为：%d，概率值为：%f" % (input_string, y_pred, confidence))
        # print(f"输入的字符串为：{input_string}，预测类别为：{y_pred}，概率值为：{confidence:.4f}")


if __name__ == '__main__':
    # main()
    test_strings = ["soaiud", "cuovihbxcvsdfqweuijethjkdfghi",
                    "dfiuhasdfasfvxcsdfwerdfeirfuth", "子丑鼠牛虎兔寅卯rtyjbsdfsdfxcvhhdfogsh",
                    "wejerwerka甲乙丙丁子丑鼠牛虎兔寅卯wedfgfdgdfgr", "戊己庚甲乙丙丁辛癸baeasuyfdfgsdrwersdgewrt"]
    predict("model.pth", "vocab.txt", test_strings)
