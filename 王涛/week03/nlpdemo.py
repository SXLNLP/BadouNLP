import json
import random
import numpy as np
from torch import nn
import torch

# 模型定义
# 定义神经网络
class NlpModel(nn.Module):
    def __init__(self,hidden_size,vector_dim,vocab):
        super().__init__()
        # 相当于向量表
        self.embedding=nn.Embedding(len(vocab),vector_dim,padding_idx=0)
        self.rnn=nn.RNN(vector_dim, hidden_size,bias=False,batch_first=True)
        self.fn=nn.Linear(hidden_size,5)
        self.loss=nn.functional.cross_entropy
    def forward(self,x,y=None):
        x=self.embedding(x)     # (batch_size,sen_len)->(batch_size,sen_len,vector_dim)
        # output（所有时间步的隐藏状态）(batch_size, sen_len, hidden_size)
        # h_n（最后一个时间步的隐藏状态）：(num_layers, batch_size, hidden_size)
        output, h_n=self.rnn(x)       # (output, h_n)
        h_n=h_n.squeeze(0)           # (num_layers, batch_size, hidden_size) -> (batch_size, hidden_size)
        y_pre=self.fn(h_n)         # (batch_size,  hidden_size) -> (batch_size,5)
        if y is not None:
            return self.loss(y_pre,y)
        else:
            return y_pre
#建立模型
def build_model(hidden_size,vector_dim,vocab):
    model = NlpModel(hidden_size,vector_dim,vocab)
    return model
# 字符表生成
def sam():
    s="awertyuiopqsdfghjklzxcvbnm"
    vocab={"pad":0}
    for idx,char in enumerate(s):
        vocab[char]=idx+1
    vocab['unk']=len(vocab)
    return vocab

# 一条数据生成
def generate_random_number_sequence(vocab, length=5):
    # 获取所有索引（包括特殊字符的索引0和27）
    all_indices = list(vocab.values())


    # 1. 固定包含索引1
    sequence = [1]
    # 2. 从剩余索引中选取`length-1`个不重复的索引
    remaining_indices = [idx for idx in all_indices if idx != 1]

    sequence += random.sample(remaining_indices, k=length-1)
    # 3. 打乱顺序
    random.shuffle(sequence)
    y=sequence.index(1)
    return sequence,y
# 建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = generate_random_number_sequence(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # 交叉熵规定的y是[1,2,3]，且是LongTensor
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
# 测试代码
def evaluate(model,vocab):
    model.eval()  # 测试模式
    x,y=build_dataset(200,vocab, 5)
    y_pre=model(x)
    y_pred=torch.argmax(y_pre,1)
    cor,wrong=0,0
    with torch.no_grad():
        for y_p,y_t in zip(y_pred,y):
            if y_p == y_t:
                cor+=1
            else:
                wrong+=1
    print(f"本次测试集的样本数是200个，预测对的样本数是{cor}个，预测正确率是{cor/(cor+wrong)}")
    return cor/(cor+wrong)
def nlp_train():
    #参数
    lr=0.001
    batch_size=20
    epoch=20
    sample_length=500
    sentence_length=5
    vocab=sam()
    model= NlpModel(10,12,vocab)
    # 优化器
    adma=torch.optim.Adam(model.parameters(),lr=lr)
    for e in range(epoch):
        watch_loss=[]
        for batch in range(sample_length//batch_size):
            # 生成数据
            x,y=build_dataset(batch_size,vocab,sentence_length)
            loss=model(x,y)  # 计算损失
            loss.backward()  # 反向传播
            adma.step()      # 参数更新
            adma.zero_grad() # 梯度清零
            watch_loss.append(loss.item())
        print(f"=========\n第{e+1}轮的损失是{np.mean(watch_loss)}")
        evaluate(model,vocab)
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
def predict(model_path, vocab_path, input_strings):
    char_dim = 12  # 每个字的维度
    # sentence_length = 5  # 样本文本长度
    hidden_size=10
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(hidden_size,char_dim,vocab)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    result = torch.softmax(result, dim=1)  # 将预测结果的维度1进行softmax
    res_p, res_cls = torch.max(result,dim=1) # 获取最大概率及其索引
    for i in range(len(input_strings)):
        print(f"=========\n输入是 {input_strings[i]} ,预测类别是{res_cls[i].item()},预测的概率是{res_p[i].item()}")
if __name__ == '__main__':
    # nlp_train()
    test_strings = ["fnvfa", "wadfg", "rwaeg", "nkwwa"]
    predict("model.pth", "vocab.json", test_strings)
