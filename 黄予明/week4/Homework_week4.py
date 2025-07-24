#Homework week4 RNN方法完成字符串的全切分
import jieba
import torch
import torch.nn as nn
import string
import re
import numpy as np

class RNN_cut_model(nn.Module):
   #模型初始化入参
    def __init__(self,vocab_size,vector_num,hidden_size,max_length):              
        super(RNN_cut_model,self).__init__() #初始化前先初始化父类，需要从当前类查找父类所以super(RNNmodel）
        self.embedding=nn.Embedding(vocab_size,vector_num,padding_idx=0)  #确保填充0的位置向量为0
        #self.batchsize=batchsize
        self.dropout = nn.Dropout()   #dropout 防止模型过于依赖某些高频字作预测，增加robust 和泛化性
        self.max_length=max_length
        self.vocab_size=vocab_size
        self.rnn=nn.RNN(vector_num,hidden_size,batch_first=True)  #RNN的input_size参数应该等于embedding层的输出维度，不是序列长度
        self.fc=nn.Linear(hidden_size,2)   
        self.loss=nn.CrossEntropyLoss() #计算多分类问题
   #调用模型入参，# 当不传入one_sentence_pred时，它默认为None  
    def forward(self,x,y_true=None):
        embedding=self.embedding(x)
        embedding = self.dropout(embedding) 
        rnn_output,_ =self.rnn(embedding) #第二个隐藏层信息丢掉
        fc_output = self.fc(rnn_output)
        
        # 数据流向  输入序列 [batch_size, seq_len] 
        #    ↓ (embedding层)
        #    [batch_size, seq_len, vector_num]
        #    ↓ (RNN层)
        #    [batch_size, seq_len, hidden_size]
        #    ↓ (全连接层)
        #    [batch_size, seq_len, 2]  # 2个类别         
        
        if y_true is None:
            # 预测模式：返回每个位置的预测结果 [batch_size, seq_len]
            predictions = torch.argmax(fc_output, dim=-1)  # 取每个位置概率最大的类别
            return predictions
        else:
            # CrossEntropyLoss的期望格式：
            # input: (N, C) - N个样本，每个样本C个类别
            # target: (N,) - N个样本的标签
            batch_size, seq_len, num_classes = fc_output.shape
            fc_output_reshaped = fc_output.view(-1, num_classes)  # [batch_size*seq_len, 2]
            y_true_reshaped = y_true.view(-1)  # [batch_size*seq_len]
            return self.loss(fc_output_reshaped, y_true_reshaped)




#构建训练样本数据集
def sample_dataset(char_dict,train_data,maxlength):

    
    #使用新建的字典，将训练数据中的每行中文转成数字列表,
    seq_train_data = []    
    with open(train_data, encoding='utf-8') as f:
        for i,line in enumerate(f):
            line_seq_train_data=[]
            # 过滤换行字符
            line = re.sub(r'\n','',line)
            for x in line:                
                if x in char_dict:
                    line_seq_train_data.append(char_dict[x])
                else:
                    line_seq_train_data.append(0)
            if len(line_seq_train_data)<maxlength:    
                line_seq_train_data.extend([0]*(maxlength-len(line_seq_train_data))) 
            else:
                line_seq_train_data=line_seq_train_data[:maxlength]        

            seq_train_data.append(line_seq_train_data) #每个元素是每一行的字符序列

    #将字符序列转成数字序列
    
    return seq_train_data
 
#构建训练的标注数据集 
def label_dataset(train_data, maxlength):
    #使用jieba的词表切词作为标注数据
    dataset_label = []

    with open(train_data, 'r', encoding='utf-8') as f:
        for line in f:
            line = re.sub(r'\n','',line)
            dataset_cut = jieba.lcut(line)
            line_labels = []
            # print(dataset_cut)
            for word in dataset_cut:
                # 为每个词创建标签序列：最后一个字符标记为1，其他字符标记为0
                word_labels = [0] * (len(word) - 1) + [1] if len(word) > 0 else []
                line_labels.extend(word_labels)
            
            # 对齐长度，补0或者截断    
            if len(line_labels) < maxlength:
                # 如果太短，在后面补0
                line_labels.extend([0] * (maxlength - len(line_labels)))
            else:
                # 如果太长，截断到指定长度
                line_labels = line_labels[:maxlength]
            
            # print(line_labels)
            dataset_label.append(line_labels)
    
    return dataset_label

#构建测试数据集
def test_Dataset(input:list[str],char_dict,max_length):     
    seq_test_data= []   
    for line in input:   
        line_seq_train_data=[]                
        line = re.sub(r'\n','',line)
        for x in line:                
            if x in char_dict:
                line_seq_train_data.append(char_dict[x])
            else:
                line_seq_train_data.append(0)
        if len(line_seq_train_data)< max_length:    
            line_seq_train_data.extend([0]*(max_length-len(line_seq_train_data))) 
        else:
            line_seq_train_data=line_seq_train_data[:max_length]        
        seq_test_data.append(line_seq_train_data) #每个元素是每一行的字符序列
    return seq_test_data    

def evaluate(test_Dataset, model, raw_input):
    """
    评估模型分词效果
    """
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():
        for index_of_raw_input, word_line in enumerate(raw_input):
            jieba_word_cut = jieba.lcut(word_line)
            test_pred = model(torch.LongTensor([test_Dataset[index_of_raw_input]]))
            test_pred = test_pred.squeeze(0).tolist()  # 转换为列表
            
            # 初始化分词结果字符串
            rnn_word_cut = ""
            
            # 根据预测结果进行分词
            for index, letter in enumerate(test_pred[:len(word_line)]):
                if index < len(word_line):
                    rnn_word_cut += word_line[index]  # 添加字符
                    if letter == 1:  # 如果是词尾
                        rnn_word_cut += "/"  # 添加分隔符
            
            print(f"原字符串: {word_line}")
            print(f"预测标签: {test_pred[:len(word_line)]}")
            print(f"jieba分词: {jieba_word_cut}")
            print(f"RNN分词: {rnn_word_cut}")


def main():
    char_path= "rnn分词/chars.txt"
    #1, 创建字典跟数字的一一对应， 从数字1开始
    char_dict = {}
    with open(char_path, encoding='utf-8') as f:
        for i,line in enumerate(f):
            char_dict[line.strip()[0]] = i+1 
    vector_num=5
    hidden_size=100
    max_length = 50
    vocab_size=4621 #char.txt里的字符总数
    
    input_data = sample_dataset(char_dict,"corpus.txt",max_length)     
    label_data = label_dataset("corpus.txt", max_length)
    model=RNN_cut_model(vocab_size,vector_num,hidden_size,max_length)
    batch_size=40
    epoch_num=50   
    batch_num=len(input_data)//epoch_num//batch_size
    learning_rate=0.001
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch in range(batch_num):
        #构造一组训练样本,每个batch从input pop 出batch size 个元素
            batch_input_data=[input_data.pop(0) for _ in range(batch_size)]
            batch_label_data=[label_data.pop(0) for _ in range(batch_size)]
            x=torch.LongTensor(batch_input_data)
            #print(x.shape)
            y_true=torch.LongTensor(batch_label_data)
            #print(y_true.shape)
            optim.zero_grad()
            loss=model(x,y_true)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))    
    torch.save(model.state_dict(), "model1.pth")
    
    model.eval()  #测试模式
    test_strings = ["我们是谁", "较前一日结算价上涨475元吨g", "大家一起去日本", "今天温度多少","复旦大学生学生宿舍在几号楼，楼梯口在哪","看看我看的位置，有好看的风景"]
    x_test=test_Dataset(test_strings,char_dict,max_length)
    #print(x_test)
    evaluate(x_test,model,test_strings)
if __name__ == "__main__":
    main()

    
    


