# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json  # 用于保存和加载词汇表

"""
基于pytorch的网络编写,实现一个网络完成一个简单nlp任务:
输入特定字符，根据字符所在位置，判断属于哪一类
"""


class TorchModel(nn.Module):  # 2.设计包含嵌入层和池化层、RNN层的神经网络模型
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding嵌入层，字符索引转向量
        # self.pool = nn.AvgPool1d(sentence_length)  # 池化层，对序列进行一维平均池化，保留整体特征
        self.rnn = nn.RNN(vector_dim, vector_dim, bias=False, batch_first=True)  # 线性层，将嵌入向量映射到分类空间
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # 字符长度为6 则有6类，但若a不在6类，需增加一类，即加1
        # self.activation = torch.sigmoid  # sigmoid归一化激活函数，用于二分类概率转换
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵，处理分类问题

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim) 嵌入层处理，字符索引转向量
        # 使用pooling方式才执行下面三步
        # x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len) 维度转置
        # x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1) 池化操作
        # x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim) 维度压缩
        # 使用rnn会输出所有输出结果和最后一个时间步的隐藏状态
        output, hidden = self.rnn(x)
        x = output[:, -1, :]  # rnn最后一层隐藏层形状（批次，时间步，隐藏层）
        y_pred = self.classify(x)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1) 激活函数，转换成概率值
        if y is not None:
            return self.loss(y_pred, y)  # 存在真实标签，即y有值，计算预测值和真实值的损失
        else:
            return y_pred  # 当y=none，则输出预测结果


# 1.词汇表构建，映射成索引：字符集随便挑了一些字，实际上还可以扩充，
# 为每个字生成一个标号，{"a":1, "b":2, "c":3...} abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 定义字符集，包括中文和英文字母
    vocab = {"pad": 0}  # 初始化词汇表，pad为填充索引，值为0,字典类型
    # 遍历字符集，为每一个字符分配索引，从1开始
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号，字典类型取值有dict[key]或dict.get(key)
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 3.显示随机生成一个样本
# 从所有字中选取sentence_length个字,反之为负样本
def build_string_with_a(vocab, sentence_length):
    # 从字典vocab的键中随机抽取不重复的sentence_length个词，组成一个列表，random.sample(str, length)
    x = random.sample(list(vocab.keys()), sentence_length)
    if 'a' in x:
        y = x.index('a')  # 获取字符a的索引位置，即第几类
    else:
        y = sentence_length  # a不存在，则使用默认值字长
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字符转换成序号，为了做embedding
    return x, y


# 3.在单个样本基础随机生成数据集，输入需要的样本数量。需要多少生成多少sample_length
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_string_with_a(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)  # 多分类，所以y转换成长整型tensor


# 建立模型，实例化TorchModel类
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 模型评估，测试代码，用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (len(y)))
    correct = 0
    wrong = 0
    with torch.no_grad():  # 不计算梯度，节省内存
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1  # 类别正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 4.主训练函数，使用Adam优化器训练，交叉熵损失，训练结束保存模型
def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 40  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.001  # 学习率
    # 建立词汇表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 开始训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):  # 训练一轮的批次数
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 调用评估模型，计算本轮的准确率
        log.append([acc, np.mean(watch_loss)])  # 记录每轮的准确率和平均损失

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    # json.dumps是将Python对象转换成Json格式字符串，writer.write写入文件
    # 参数ensure_ascii=False是不强制使用 ASCII 编码，保留非 ASCII 字符（如中文）
    # 参数indent=2美化格式，每行缩进 2 个空格
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 5.用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ['qhsoegsbau', 'dafchiegsm', 'poshtwrxbj', 'nesvbyihnz']
    predict("model.pth", "vocab.json", test_strings)
