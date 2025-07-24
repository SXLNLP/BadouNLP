'''
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
使用交叉熵实现一个多分类任务，
规律：x是一个5维向量，样本总共分为五类，五维随机向量最大的数字在哪维就属于哪一类。
'''

# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.layer=nn.Linear(input_size,5) #线性层
        self.ln=nn.LayerNorm(5)
        self.dropout=nn.Dropout(p=0.3)
        self.activate=nn.functional.sigmoid
        self.loss=nn.functional.cross_entropy # loss函数采用交叉熵损失

    def forward(self,x,y=None):
        x=self.ln(x)
        x=self.dropout(x)
        y_pred=self.layer(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

def build_sample():
    x=np.random.random(5)
    max_index=np.argmax(x)
    y=np.zeros(5)
    y[max_index]=1
    return x,y

def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num=100
    x,y=build_dataset(test_sample_num)
    correct,wrong=0,0
    with torch.no_grad():
        y_pred=model(x)
        for y_p,y_t in zip(y_pred,y):
            if torch.argmax(y_p)==torch.argmax(y_t):
                correct+=1
            else:
                wrong+=1
    print('正确预测个数: %d, 正确率: %f' %(correct,correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num=200
    batch_size=20
    train_sample=5000
    input_size=5
    learning_rate=0.0001

    model=TorchModel(input_size)
    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
    log=[]
    train_x,train_y=build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch_index in range (train_sample // batch_size):
            x=train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y=train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss=model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print('=======\n第%d轮平均loss: %f' %(epoch+1,np.mean(watch_loss)))
        acc=evaluate((model))
        log.append([acc,float(np.mean(watch_loss))])
    torch.save(model.state_dict(),'model.bin')
    print(log)
    plt.plot(range(len(log)),[l[0] for l in log], label='acc')
    plt.plot(range(len(log)),[l[1] for l in log], label='loss')
    plt.legend()
    plt.show()
    return

def test(input_vec):
    input_size=len(input_vec[0])
    model_path='model.bin'
    model=TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result=model.forward(torch.FloatTensor(input_vec))
    for vec,res in zip(input_vec,result):
        max_prob, y_class=torch.max(res,dim=0)
        print('输入: %s, 预测类别: %d, 概率值: %f' %(vec. y_class, max_prob))
if __name__=='__main__':
    main()

    test(test_vec)
