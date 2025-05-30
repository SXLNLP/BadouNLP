import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        '''
        RNN 的默认设置是 batch_first=False，这导致它把序列长度（5）当作了 
        batch size，而把实际的 batch size（20）当作了序列长度。
        通过设置 batch_first=True，我们告诉 RNN 正确的维度顺序，这样它就能正确处理所有 20 个样本了。
        '''
        self.classify = nn.RNN(vector_dim, 5, batch_first=True, bias=False)     #RNN层，设置batch_first=True
        self.activation = nn.Softmax()     #sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        tmpx,h = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*20 20*1 -> 3*1
        print('tmpx shape:', tmpx.shape)
        print('h shape:', h.shape)
        # print('tmpx:', tmpx)
        # print('h:', h)
        h = h.squeeze(0)  # 从 [1, 20, 5] 变成 [20, 5]
        if y is not None:
            return self.loss(h, y)   #预测值和真实值计算损失
        else:
            return self.activation(h)  #没有真实标签时，使用自己的softmax做预测
        

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}   
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice([char for char in vocab.keys() if char not in ['a']]) for _ in range(sentence_length)]
    aid =random.randrange(0, sentence_length-1)
    x[aid] =  "a"
    # print('before ',x)
    # print('aid ',aid)
    x = [vocab.get(word, vocab['unk']) for word in x]
    # print('after ',x)
    return x, aid


# build_sample(build_vocab(),5)
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

def evaluate(model, vocab, sample_length): #这里的model代表所有层的总和的整体，所以这里的modex（x）是经过所有层的预测，
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    # 统计每个类别的样本数
    class_counts = torch.bincount(y.squeeze())
    print("本次预测集中各类别样本数：", class_counts.tolist())
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if y_p.item() == y_t.item():  # 使用 item() 获取标量值进行比较
                correct += 1
            else:
                wrong += 1
    print("错误预测个数：%d, 错误率：%f"%(wrong, wrong/(correct+wrong)))
    print("错误样本：")
    for y_p, y_t in zip(y_pred, y):  
        if y_p.item() != y_t.item():  
            print("预测值：%d，真实值：%d" % (y_p.item(), y_t.item()))
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 50        #训练轮数
    batch_size = 10       #每次训练样本个数
    train_sample = 100    #每轮训练总共训练的样本总数
    char_dim = 5         #每个字的维度
    sentence_length = 5   #样本文本长度
    learning_rate = 0.005 #学习率
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
            print('main x ',x)
            print('main y ',y)
            optim.zero_grad()    #梯度归零
            print('main x shape ',x.shape)
            print('main y shape ',y.shape)
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


if __name__ == "__main__":
    main()
