#构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。
import torch.nn as nn
import torch
import numpy as np

num_embeddings = 28 #nlp中，应该等于字典索引长度
embedding_dim = 5 #每个字符向量化后的向量维度 ，不固定，人为定义

#构建词表
word_dict ={
  "pad": 0,
  "a": 1,
  "b": 2,
  "c": 3,
  "d": 4,
  "e": 5,
  "f": 6,
  "g": 7,
  "h": 8,
  "i": 9,
  "j": 10,
  "k": 11,
  "l": 12,
  "m": 13,
  "n": 14,
  "o": 15,
  "p": 16,
  "q": 17,
  "r": 18,
  "s": 19,
  "t": 20,
  "u": 21,
  "v": 22,
  "w": 23,
  "x": 24,
  "y": 25,
  "z": 26,
  "unk": 27
}

#将字符串转换为序列
def str_to_sequence(string, word_dict):
    seq= [word_dict[w] if w in word_dict else word_dict["unk"] for w in string]
    print(seq)
    return seq

#1：  embedding layer 每个字符转化成同维度向量,构建embedding层,将字符向量张量化，然后embedding 
def embedding(seq):
    input = torch.tensor(seq, dtype=torch.long)
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
    # print(f"input: {input}")
    embedded_input = embedding_layer(input)
    # print(f"embedded_input: {embedded_input}")
    return embedded_input

#2: RNN
class Muti_classification(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,setence_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) #文本矩阵化   
        self.rnn = nn.RNN(embed_dim, hidden_size,batch_first=True) #when batch_first=True,输入数据形状默认为batch size, set_length,vector dim(input size)
       # self.pool=nn.AvgPool1d(kernel_size=embed_dim)   # 降低维度，一个单词的特征提取，5维转成1维,损失维度会导致预测不准
        self.fc = nn.Linear(hidden_size, setence_length) # 考虑加一层线性层将RNN输出映射回类别空间，10分类就是10
        self.activation = nn.Softmax(dim=1)     #sigmoid归一化函数用于二分类，多分类用softmax
        self.loss = nn.CrossEntropyLoss() #多分类问题

    def forward(self, x,y_true=None):
        embedded = self.embedding(x)
        # print(embedded.size())
        # print(f"embedded: {embedded}")
        rnn_output, _ = self.rnn(embedded)
        rnn_output = rnn_output[:, -1, :]  # 只取最后一个时间步的输出
        fc_output = self.fc(rnn_output) 
        # print(rnn_output.size()) #确保交叉熵传参数形状最后一个是类别数量
        y_pred=self.activation(fc_output)      #sigmoid归一化函数
        if y_true is not None:

            #y_true = y_true.unsqueeze(1)  # 添加一个维度 [13] -> [13, 1]
            return self.loss(y_pred, y_true)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果
    
    

#3: pooling 
def pooling(embedded_input):
    # 最大池化
    max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    x = torch.tensor(embedded_input).unsqueeze(0)
    y_max = max_pool(x)
    # y_max: tensor([[[3., 4., 6.]]])

    # 平均池化
    avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
    y_avg = avg_pool(x)
    # y_avg: tensor([[[2., 3., 5.5]]])

#build sample
def build_sample(num_sample,vocab,setence_length):
    sample_list=[]
    for i in range(num_sample):
        l=[]
        for x in range(setence_length):            
            l.append(np.random.choice(list(vocab.values())[1:]) )
        if 1 in l: 
            sample_list.append(l)
    return sample_list    #多维列表张量化  

#build test result of sample using one hot vector
def find_a_position_with_onehot(build_sample):
    y_true = []
    for i in range(len(build_sample)):
        for x,y in enumerate(build_sample[i]):
          if y==1:
              y_true.append(x)
              break 
    return y_true

def find_max(v):
    a = []
    vector = np.zeros(len(v))
    for x in range(len(v)):
        if a == [] or v[x] > v[a[-1]]:
            a.append(x)
    return a[-1]

def main():
    batch_size=100
    embed_dim=5
    num_epoch=15
    setence_length=6
    hidden_size=10 
    text_model=Muti_classification(batch_size, embed_dim, hidden_size,setence_length) 
    learning_rate = 0.005 #学习率
    
    # 选择优化器Adam根据历史数据调节学习率
    optim = torch.optim.Adam(text_model.parameters(), lr=learning_rate)   
    batch_num=500 #一轮做几个batch

    for i in range(num_epoch):
        text_model.train()
        watch_loss=[] #每轮loss,最后求平均
        
        for batch in range(batch_num):
            one_sample=build_sample(batch_size,word_dict,setence_length)  #每轮的样本
            #print(len(one_sample))
            y_pred=torch.tensor(one_sample, dtype=torch.long) #       
            y_true=torch.tensor(find_a_position_with_onehot(one_sample), dtype=torch.long) #
            
            loss=text_model(y_pred,y_true)
            #反向传播时梯度归0
            optim.zero_grad()
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
        
        text_model.eval() #每轮的准确率测试不用train mode
        test_sample=build_sample(30,word_dict,setence_length) 
        test_result=text_model(torch.tensor(test_sample, dtype=torch.long))
        # print(test_sample)
        # print(test_result)
        correct_count=0
        for x in range(len(test_result)):
            target_index=find_max(test_result[x])
            # print(target_index)
            if test_sample[x][target_index]==1:
               correct_count+=1 
        accuracy=correct_count/len(test_result)
        print(f"准确率: {accuracy}")
    #保存模型
    torch.save(text_model.state_dict(), "model.pth")   
        

        



if __name__ == "__main__":   
   
    main()
   
    
    

    
   






