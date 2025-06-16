#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import copy
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(vector_dim, 64, bias=True, batch_first=True)
        self.classify = nn.Linear(64, sentence_length)     #线性层
        self.loss = nn.CrossEntropyLoss()  #loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        _, x = self.rnn(x)                         #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, 64),(1, batch_size, 64)
        x = x.squeeze(0)                           #(1, batch_size, 64) -> (batch_size, 64)
        y_pred = self.classify(x)                  #(batch_size, 64) -> (batch_size, sen_len)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return torch.softmax(y_pred,dim=1)                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

# 为字符串填充padding
def padding(x, max_sentence_length):
    return x + [0] * (max_sentence_length - len(x)) \
        if len(x) < max_sentence_length else x[:max_sentence_length]

#随机生成一个样本
#构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置
def build_sample(vocab, max_sentence_length):

    # 去掉pad符号，方便后面构造字符串
    voc = copy.deepcopy(vocab)
    del voc["pad"]

    #构造随机长度的句子
    sentence_length = random.randint(0,max_sentence_length-1)

    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(voc.keys())) for _ in range(sentence_length)]

    #在x的随机位置插入字符a
    x.insert(random.randint(0, len(x)), "a")

    #第一次出现a的位置是y的值
    y = x.index("a")

    x = [voc.get(word, voc['unk']) for word in x]   #将字转换成序号，为了做embedding
    x = padding(x, max_sentence_length)

    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
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
@torch.no_grad
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(500, vocab, sample_length)   #建立200个用于测试的样本
    for i in range(sample_length):
        pass
        # print(f"预测集中共有{int(y.sum()[i].item())}个第{i}类样本")
    correct, wrong = 0, 0

    y_pred = model(x)  # 模型预测 model.forward(x)
    val_loss = model(x, y).item()
    for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
        if torch.argmax(y_p) == int(y_t):
            correct += 1  # 分类正确
        else:
            wrong += 1  #分类错误
    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong)}, loss: {val_loss}")
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 2000    #每轮训练总共训练的样本总数
    char_dim = 32         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.0025 #学习率
    # 建立字表
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
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
@torch.no_grad
def predict(model_path, vocab_path, input_strings):
    char_dim = 32  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))            #加载训练好的权重
    x = []
    for input_string in input_strings:
        input_string = [vocab.get(word, vocab['unk']) for word in input_string]  # 将字转换成序号，为了做embedding
        input_string = padding(input_string, sentence_length)  #填充string
        x.append(input_string)
    model.eval()   #测试模式

    result = model.forward(torch.LongTensor(x))  #模型预测
    classify = torch.argmax(result, dim=1)

    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}, 预测类别：{round(float(classify[i]))}, 概率值：{result[i][classify[i]]}") #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["fn你是eaaaa", "az你d大家ag", "rqawdeg他我们他他", "aan我kbfu她sdibuiswww"]
    predict("model.pth", "vocab.json", test_strings)
    '''
    输入：fn你是eaaaa, 预测类别：5, 概率值：0.9680255055427551
    输入：az你d大家ag, 预测类别：0, 概率值：0.9992982149124146
    输入：rqawdeg他我们他他, 预测类别：2, 概率值：0.9974706172943115
    输入：aan我kbfu她sdibuiswww, 预测类别：0, 概率值：0.9981445074081421
    '''
    # == == == == =
    # 第1轮平均loss: 1.033700
    # 正确预测个数：498, 正确率：0.996, loss: 0.08389697223901749
    # == == == == =
    # 第2轮平均loss: 0.036777
    # 正确预测个数：500, 正确率：1.0, loss: 0.016048777848482132
    # == == == == =
    # 第3轮平均loss: 0.011238
    # 正确预测个数：500, 正确率：1.0, loss: 0.008012541569769382
    # == == == == =
    # 第4轮平均loss: 0.006241
    # 正确预测个数：500, 正确率：1.0, loss: 0.005082570482045412
    # == == == == =
    # 第5轮平均loss: 0.004175
    # 正确预测个数：500, 正确率：1.0, loss: 0.0037285122089087963
    # == == == == =
    # 第6轮平均loss: 0.003180
    # 正确预测个数：500, 正确率：1.0, loss: 0.002465466968715191
    # == == == == =
    # 第7轮平均loss: 0.002381
    # 正确预测个数：500, 正确率：1.0, loss: 0.0018220038618892431
    # == == == == =
    # 第8轮平均loss: 0.001799
    # 正确预测个数：500, 正确率：1.0, loss: 0.0018091415986418724
    # == == == == =
    # 第9轮平均loss: 0.001524
    # 正确预测个数：500, 正确率：1.0, loss: 0.0012909352080896497
    # == == == == =
    # 第10轮平均loss: 0.001262
    # 正确预测个数：500, 正确率：1.0, loss: 0.0011445903219282627

