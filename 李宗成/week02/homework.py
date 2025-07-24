
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        #线性层
        self.linear = nn.Linear(input_size, output_size)

        #激活函数
        self.activation = nn.Sigmoid()

        #损失函数 交叉熵
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        #batch_size input_size => batch_size output_size
        x = self.linear(x)

        #预测结果
        y_pred = self.activation(x)

        if y is not None:
            #计算损失
            return self.loss(y_pred, y)
        else:
            #输出预测结果
            return y_pred
        
#样本生成
def build_dataset(single_sample_num, total_sample_num):
    #生成样本
    X = []
    Y = []
    for i in range(total_sample_num):
        x = np.random.random(single_sample_num)

        #找出最大值并记录下标
        max_index = np.argmax(x)
        y = np.zeros(single_sample_num)
        y[max_index] = 1

        X.append(x)
        Y.append(y)

    return torch.FloatTensor(X), torch.FloatTensor(Y)

#处理数据
def normalize(data_list):
    max_index = np.argmax(data_list)
    data_list = torch.zeros(len(data_list))
    data_list[max_index] = 1
    return data_list


def evaluate(model, sample_num, total_sample_num):
    #计算正确率
    model.eval()
    x, y = build_dataset(sample_num, total_sample_num)
    
    with torch.no_grad():
        #模型预测
        y_pred = model(x)
        
        correct, wrong = 0, 0

        for y_p, y_t in zip(y_pred, y):
            y_p = normalize(y_p)

            if y_p.equal(y_t):
                correct += 1
            else:
                wrong += 1
        
        print('正确率：', correct / (correct + wrong))
        return correct / (correct + wrong)
    
def main():
    #超参数
    epoch_num = 100
    batch_size = 20
    train_num = 5000
    input_size = 5
    output_size = 5
    learning_rate = 0.001

    #建立模型
    model = TorchModel(input_size, output_size)
    #优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #创建训练集
    x, y = build_dataset(input_size, train_num)

    log = []
    #训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(int(train_num / batch_size)):
            x_batch = x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_batch = y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x_batch, y_batch)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        
        print('第%d轮平均loss:%f' % (epoch + 1, np.mean(watch_loss)))
        accuracy = evaluate(model, input_size, train_num)
        log.append([accuracy, float(np.mean(watch_loss))])

    #保存模型
    torch.save(model.state_dict(), 'model.pth')

    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label='accuracy')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    plt.show()
    return

def predict(input_data):
    #加载模型
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_data))
    
    for vec, res in zip(input_data, result):
        vec = normalize(vec)
        res = normalize(res)
        print('输入：%s, 输出：%s' % (vec, res))
if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.30797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.49349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict(test_vec)
    
