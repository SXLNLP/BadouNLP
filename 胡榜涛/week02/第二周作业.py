import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
'''
一个5分类的深度学习demo
'''
class Mymodule(nn.Module):
    def __init__(self,input_size,output_size):
        super(Mymodule,self).__init__()
        self.linear1 = nn.Linear(input_size,64)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(64,output_size)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
    def forward(self,x):
        x=self.linear1(x)
        y_pred=self.sigmoid(x)
        y_pred=self.linear2(y_pred)
        return y_pred

    def loss_eva(self,y_true,y_pred):
        print("first",y_true)
        y_true=torch.argmax(y_true,dim=1)
        y_true=torch.LongTensor(y_true)
        print("two",y_true)
        return self.loss(y_pred,y_true)

def build_sample_data(sample_size):
    X=[]
    Y=[]
    for i in range(sample_size):
        x=np.random.random(5)
        y=np.zeros(5)
        x_max_index=np.argmax(x)
        y[x_max_index]=1.0
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)#np.array会将元素转换为相同类型，这里不影响

def evaluate(module):
    module.eval()
    eva_sample_num=100
    x,y=build_sample_data(eva_sample_num)
    correct=0
    wrong=0
    with torch.no_grad():
        y_pred=module.forward(x)
        for y_t,y_p in zip(y,y_pred):
            if np.argmax(y_p)==np.argmax(y_t):
                correct+=1
            else:
                wrong+=1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num=30
    sample_size=1000
    batch_size=20
    module=Mymodule(5,5)
    x_t,y_t=build_sample_data(sample_size)
    log=[]
    for epoch in range(epoch_num):
        module.train()
        epoch_loss=[]
        for batch_index in range(sample_size//batch_size):
            x_true=x_t[batch_index*batch_size:(batch_index+1)*batch_size]
            y_true=y_t[batch_index*batch_size:(batch_index+1)*batch_size]
            y_pred=module.forward(x_true)
            loss=module.loss_eva(y_true,y_pred)
            loss.backward()
            module.optimizer.step()
            module.optimizer.zero_grad()
            epoch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(epoch_loss)))
        acc = evaluate(module)  # 测试本轮模型结果
        log.append([acc, float(np.mean(epoch_loss))])
    torch.save(module.state_dict(),'model_path')
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return

def test(model_path):
    module=Mymodule(5,5)
    module.load_state_dict(torch.load(model_path,weights_only=True))
    acc = evaluate(module)
    print("训练完后测试的正确率为：",acc)



if __name__ == '__main__':
    main()
    test("model_path")

