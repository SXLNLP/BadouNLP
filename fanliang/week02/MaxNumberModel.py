import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MaxNumberModel(nn.Module):
    def __init__(self,inputsize):
        super(MaxNumberModel,self).__init__()
        self.linear = nn.Linear(inputsize,5)
        self.activefunction = nn.Softmax()
        self.lossfunction = nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        linout = self.linear(x)
        if y is not None:
            return self.lossfunction(linout,y)#CrossEntropyLoss中含有Softmax所以如果是训练模式，则不需要再进行Softmax
        else:
            return self.activefunction(linout) 

def build_sample():
    x = np.random.random(5)
    x_tensor = torch.FloatTensor(x)  # 将numpy数组转换为torch张量
    maxindex = torch.argmax(x_tensor)
    return x, maxindex

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本" % test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            pred_index = torch.argmax(y_p)  
            if pred_index == y_t:
                correct += 1
            else:
                print("evaluate error: y_p: {}, y: {}, pred_index: {}".format(y_p, y_t, pred_index))
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = MaxNumberModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() #设置为训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
