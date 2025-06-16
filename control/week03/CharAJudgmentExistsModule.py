
# 将a出现的位置信息作为分类类别，对任意输入的句子进行分类
'''
修改了池化层，激活函数，还有损失函数。发现部分修改 损失函数不会收敛。或损失函数一直为 0 。或无法匹配使用的情况。
字符串的分类和 a的位置信息是强相关的。a的位置和a的分类是一种线性的回归问题。
那么，通过 rnn网络，在最后一个
'''
import numpy as np
import torch
import torch.nn as nn
import random
import json

class JudgeExistsChar(nn.Module):

    # vocabsize : 字典表的大小，用于说明 embedding 要随机的 向量的个数
    # char_dim : 生成的字字向量的维度
    def __init__(self, vocabsize, char_dim, sentence_length):
        super(JudgeExistsChar, self).__init__()
        # 创建embedding层， padding_idx  不够句子长度，则用 0 补齐
        self.embedding = nn.Embedding(vocabsize, char_dim, padding_idx=0) # sentence_length *  vocabsize
        self.rnn_layer = nn.RNN(char_dim, sentence_length, bias= True, batch_first=True) # 循环神经网络，会传递序列顺序信息到最后一个字符上。
        self.linear_layer = nn.Linear(sentence_length, 1) # sentenct_length 的长度，就是a会插入的位置的种类数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差
    def forward(self, x, y=None):
        # 将字符 转成 向量   batch_size  *  sentence_length  =  (20 * 20)
        embding_x = self.embedding(x)   #(batch_size, sentence_length) -> (batch_size, sentence_length, char_dim)
        y_pred = self.rnn_layer(embding_x)   #(batch_size, sentence_length, char_dim) -> (batch_size, sentence_length, char_dim)
        y_pred = self.linear_layer(y_pred[0][:, -1, :]) # (batch_size, char_dim) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred,  y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果



class SampleData:
    def __init__(self, sentence_length ):
        self.vocab = self.build_vocabulary()
        self.sentence_length = sentence_length
    def build_vocabulary(self): # 创建词汇表， 后边会经过embedding 层， 一个字符对应一个向量
        chars = "你我他defghijklmnopqrstuvwxyza"  # 字符集
        vocab = {"pad": 0}
        for index, char in enumerate(chars):
            vocab[char] = index + 1  # 每个字对应一个序号
        vocab['unk'] = len(vocab)  # 26
        return vocab
    def build_sample(self): # 创建一条样本
        vocab_keys  = list(self.vocab.keys())
        #删除 a这个 key，并生成新的 list
        no_a_cocab = [char for char in vocab_keys if char != "a"]
        x = [random.choice(no_a_cocab) for _ in range(self.sentence_length - 1)]
        a_index = random.randint(0, self.sentence_length - 1) # 生成a插入的位置
        x.insert(a_index, "a")
        return x,a_index  #字符串长度：self.sentence_length , a_index: a在句子中的位置

    def build_dataset(self, sample_length): # 创建数据集, sample_length: 样本个数
        x_dataset = []
        y_dataset = []
        for i in range(sample_length):
            x, y = self.build_sample()
            x = [self.vocab.get(word, self.vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
            x_dataset.append(x)
            y_dataset.append([y])
        return torch.LongTensor(x_dataset), torch.FloatTensor(y_dataset)

    def getVocab(self):
        return self.vocab

class Test:

    def print_test_sample_distribution(self, y):
        print("测试案例-a分类频率分布为：")
        unique_values, counts = np.unique(y.numpy(), return_counts=True)
        # 打印结果
        for val, cnt in zip(unique_values, counts):
            print(f"分类值 {val} 出现 {cnt} 次")
    def evaluate(self, model, sampleData, test_sample_length):
        model.eval()
        x, y = sample_data.build_dataset(test_sample_length)  # 建立200个用于测试的样本
        print("测试数据集共有%d个样本" % test_sample_length)
        self.print_test_sample_distribution(y)
        correct, wrong = 0, 0
        with torch.no_grad():
            y_pred = model(x)  # 模型预测
            for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
                if int(round(y_p.numpy()[0])) == int(y_t.numpy()[0]) :
                    correct += 1  # 分类正确
                else:
                    wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)


def predict(model_path, input_strings, y_trues ):
    chart_dim = 20  # 每个字的维度
    sentence_length = 20  # 样本文本长度

    model = JudgeExistsChar(len(sample_data.getVocab()), chart_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    vocab = sample_data.getVocab()
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("模拟输入：%s, 预测类别：%d, 真实类别：%d" % (input_string, round(float(result[i])), int(y_trues[i]))) #打印结果


def main():
    sample_num = 1000 # 样本数量
    sentence_length = 20 #  句子长度
    learning_rate = 0.001 # 学习率
    epoch_num = 1000 # 训练轮数
    batch_size = 20 # 每个批次训练多少数据
    chart_dim = 20 # 字向量的维度
    sample_data = SampleData(sentence_length =  sentence_length)
    model =  JudgeExistsChar(len(sample_data.getVocab()), chart_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(sample_num / batch_size)):
            x,y = sample_data.build_dataset(batch_size)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        if np.mean(watch_loss) < 0.001:
            break
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    # 阻塞，等待输入
    input("Press any key to test...")
    Test().evaluate(model, sample_data, 200)
    print("测试结束！")
if  __name__ == "__main__":

    main() # model training

    input("Press any key to predict ...")
    sentence_length = 20  # 样本文本长度
    sample_data = SampleData(sentence_length=sentence_length)
    for i in range(10):
        x, y = sample_data.build_sample()
        predict(model_path="model.pth", input_strings=[x],y_trues = [y])

