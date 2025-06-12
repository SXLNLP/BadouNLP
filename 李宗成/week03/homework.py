import torch
import torch.nn as nn
import numpy as np
import json
import random

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
构造随机包含a的字符串,使用rnn进行多分类,类别为a第一次出现在字符串中的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sentence_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.RNN(embedding_dim, 128, bias=True, batch_first=True)
        self.linner  = nn.Linear(128, sentence_length)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x) 
        _, x = self.classifier(x)
        x = x.squeeze(0)
        y_pred = self.linner(x)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=1)

#  建立字表   
def build_vocab(sentences) :
    vocab = {"pad": 0}
    for sentence in sentences :
        for word in sentence :
            if word not in vocab :
                vocab[word] = len(vocab)
    vocab["unk"] = len(vocab)
    return vocab

#  生成样本
def generate_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x[random.randint(0, sentence_length - 1)] = 'a'  # 强制插入 'a'

    y = x.index('a')

    x = [vocab.get(char, vocab['unk']) for char in x]

    return x, y

#  生成训练数据
def generate_data(sample_length, vocab, sentence_length) :
    dateset_x, dateset_y = [], []
    
    for i in range(sample_length) :
        x, y = generate_sample(vocab, sentence_length)
        dateset_x.append(x)
        dateset_y.append(y)

    return torch.LongTensor(dateset_x), torch.LongTensor(dateset_y)

# 训练
def evaluate(model, vocab, sample_length) :
    model.eval()
    x, y = generate_data(200, vocab, sample_length)
    correct, wrong = 0, 0
    with torch.no_grad() :
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y) :
            if torch.argmax(y_p) == int(y_t) :
                correct += 1
            else :
                wrong += 1
    
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
    
def main() :
    epoch_num = 20          # 训练轮数
    batch_size = 20         # 每次训练样本个数
    train_data_size = 500   # 训练数据量
    char_dim = 20           # 字向量维度
    sentence_length = 5     # 句子长度
    learning_rate = 0.003   # 学习率

    vocab = build_vocab(['abcde', 'ghijk', 'opqrs', 'uvwxyz'])
    model = TorchModel(
        vocab_size=len(vocab),
        embedding_dim=char_dim,
        sentence_length = sentence_length
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num) :
        model.train()
        watch_loss = []
        for _ in range(train_data_size // batch_size) :
            x, y = generate_data(batch_size, vocab, sentence_length)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("第%d轮平均loss: %.5f" % (epoch + 1, np.mean(watch_loss)))
        accuracy = evaluate(model, vocab, sentence_length)
        log.append([accuracy, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModel(
        vocab_size=len(vocab),
        embedding_dim=char_dim,
        sentence_length = sentence_length
    )     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测

    classify = torch.argmax(result, dim=1)
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(classify[i])), classify[i])) #打印结果


if __name__ == "__main__" :
    main()
    # test_strings = ["xiasd", "apspw", "mcxam", "pppaa", "aaaaa"]
    # predict("model.pth", "vocab.json", test_strings)
