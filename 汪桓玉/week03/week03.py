import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os

# 构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

class TorchModel(nn.Module):
    """
    一个基于RNN的神经网络模型，用于预测字符'a'在字符串中首次出现的位置
    
    该模型包含以下组件:
    - embedding层: 将输入字符转换为向量表示
    - RNN层: 处理序列数据
    - Dropout层: 防止过拟合
    - 线性层: 将RNN输出映射到预测类别
    
    参数:
        vector_dim (int): 词向量维度
        hidden_size (int): RNN隐藏层大小
        vocab (dict): 词汇表字典
    
    输入:
        x: 输入序列张量
        y: 目标标签张量(可选)
    
    输出:
        如果提供y，返回损失值
        否则返回预测概率分布
    """
    
    def __init__(self,vector_dim,hidden_size,vocab):
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(vector_dim,hidden_size,batch_first=True)
        self.relu = nn.ReLU()
        self.loss = nn.functional.cross_entropy
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_size,10)
        # self.linear2 = nn.Linear(15,len(vocab))
    def forward(self,x,y=None):
        x = self.embedding(x)  
        x = self.dropout(x)
        _,x = self.rnn(x)
        x = x.squeeze(0) # 压缩维度
        x = self.linear1(x)
        # x = self.relu(x)
        if y is not None:
            loss = self.loss(x,y)
            return loss
        else:
            y_pred = self.softmax(x)
            return y_pred

# 建立字典
def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {
        "pad":0
        }
    for i ,char in enumerate(chars):
        vocab[char] = i+1
    vocab['unk'] = len(vocab)
    return vocab

def create_train_sample(num,vocab):
    max_length = 10  # 固定最大长度为10
    batch_x = []
    batch_y = []
    words = []
    for _ in range(num):
        sentences_length_max = 10
        sentences_length_min = 3
        sentences_length = np.random.randint(sentences_length_min,sentences_length_max+1)  
        # 生成不包含'a'的随机字符串
        chars = [k for k in vocab.keys() if k != 'a' and k != 'pad' and k != 'unk']  # 排除'a'和特殊字符
        x = [random.choice(chars) for _ in range(sentences_length)]
        # 在随机位置插入'a'
        a_position = np.random.randint(0, sentences_length)
        x[a_position] = 'a'
        # 补齐到固定长度
        if len(x) < max_length:
            x = x + [vocab['pad']] * (max_length - len(x))
        words.append(x)
        x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
        batch_x.append(x)
        batch_y.append(a_position)
   
    return torch.LongTensor(batch_x), torch.LongTensor(batch_y),words

def calculate_result(model,x,y,words=None):
    correct = 0
    error = 0
    y_pred = model(x)
    y_pred = torch.argmax(y_pred,dim=1)
    if words is not None:
        for y_p,y_t,w in zip(y_pred,y,words):
            print(f'预测结果：{y_p} 实际结果：{y_t} 字符串：{w}')
            if y_p == y_t:
                correct += 1
            else:
                error += 1
    else:
        for y_p,y_t in zip(y_pred,y):
            if y_p == y_t:
                correct += 1
            else:
                error += 1
    print("总数：%d 正确预测个数：%d, 错误个数：%d 正确率：%f" % (correct+error, correct, error, correct / (correct + error)))  
    return correct, correct / (correct + error)

def evaluate(model,vocab):
    model.eval()
    test_sample = 100
    x,y,_ = create_train_sample(test_sample,vocab)
    with torch.no_grad():
        _,acc = calculate_result(model,x,y)
    return acc

def predict(model_src,vocab_src):
    input_size = 10
    hidden_size = 15
    # 读取vocab_src
    with open(vocab_src, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    model = TorchModel(input_size,hidden_size,vocab)
    model.load_state_dict(torch.load(model_src,weights_only=True))
    model.eval()
    test_sample = 1000
    x,y,words = create_train_sample(test_sample,vocab)
    with torch.no_grad():
        print(f'总预测样本：{test_sample}')
        calculate_result(model,x,y,words)

def main():
    # 配置参数
    epoch_num = 10 # 训练轮数
    batch_size = 20 # 每次训练个数
    train_sample = 5000 # 每轮样本训练数
    learning_rate = 0.001 # 学习率
    input_size = 10 #输入维度/每次输入字符长度
    hidden_size  = 15 #隐藏层维度
    #建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(input_size,hidden_size,vocab)
    # 建立优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # 画图
    log = []
    
    for epcho in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample//batch_size):  # 使用整除//
            x_train,y_train,words = create_train_sample(batch_size,vocab)
          
            loss = model(x_train,y_train)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epcho+1, np.mean(watch_loss)))
        acc = evaluate(model,vocab)
        log.append([acc, np.mean(watch_loss)])

    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建保存路径
    save_path = os.path.join(current_dir, 'week03model.bin')
    # 保存词表
    vocab_path = os.path.join(current_dir, 'vocab.json')
    writer = open(vocab_path, "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    # 保存模型
    torch.save(model.state_dict(),save_path)
    # 画图

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建保存路径
    path_model = os.path.join(current_dir, 'week03model.bin')
    path_dict = os.path.join(current_dir, 'vocab.json')
    # 检查模型文件是否存在
    if not os.path.exists(path_model):
        print("模型文件不存在，开始训练...")
        main()
    else:
        print("找到模型文件，开始预测...")
    
    predict(path_model,path_dict)
    