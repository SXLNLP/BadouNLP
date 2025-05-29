'''
@Project ：DeepLearning 
@File    ：Homework_week02.py
@IDE     ：PyCharm 
@Author  ：Coin
@Date    ：2025/5/22 0:29 
'''
import torch
import torch.nn as  nn
import numpy as np
import matplotlib.pyplot as plt

#定义模型
class TorchModel(nn.Module):
    def __len__(self,input_size):
        super(TorchModel,self).__init__()
        #线性层
        self.liner = nn.Linear(input_size,5)
        #归一化
        self.activation = nn.Sigmoid()
        #loss函数：使用交叉熵,因为是分类任务
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        y_pred = self.liner(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return self.activation(y_pred)


def build_sample():
    x =np.random.random(5)
    max_value = np.max(x)
    max_index = np.argmax(x)
    return x,max_index,max_value

#制作数据集
def build_dataset(total_sample_num):
    X =[]
    Y =[]
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

#测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y =build_dataset(test_sample_num)
    ylist = y.numpy().tolist()
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.argmax(y_pred,dim=1)
        corrent = (y_pred == y ).sum().item()
        accuracy = corrent / len(y)
        print("准确率为：",accuracy)
        return accuracy


def main():
    #参数配置
    epoch_num = 50
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001
    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x ,train_y = build_dataset(train_sample)

    print("训练数据集：")
    print(train_x)
    print(train_y)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample//batch_size):
            x=train_x[batch_index*batch_size:(batch_index+1)* batch_size]
            y=train_y[batch_index*batch_size:(batch_index+1)* batch_size]
            loss = model.forward(x,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
            print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
            acc = evaluate(model)  # 测试本轮模型结果
            log.append([acc,float(np.mean(watch_loss))])
    #保存模型
    torch.save(model.state_dict(),"CorssEL.bin")



#预测
def predict(model_path,input_vec):
    input_size= 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    count=0
    for i in np.argmax(result.numpy(),axis=1):
        print("第%d个样本预测结果：%d" % (count+1,i))
        count += 1

if __name__ == '__main__':
    main()
    # test_vec = [[25, 3, 7, 2, 22],
    #             [45, 2, 222, 1, 109],
    #             [0.00797868, 0.682528, 0.1365847, 0.345372, 0.1392],
    #             [0.093776, 0.594169, 0.9259291, 0.467412, 0.158894],
    #             [2, 4, 8, 12, 10]]
    #predict("CorssEL.bin",test_vec)
    predict("CorssEL.bin")
