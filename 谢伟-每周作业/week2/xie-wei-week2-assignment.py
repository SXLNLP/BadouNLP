# 使用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # linear layer
        self.linear = nn.Linear(input_size, 5) # the output is a 5-dimension matrix
        # activation: softmax
        self.activation = nn.Softmax() #nn.CrossEntropyLoss中已经包含了softmax，故这里不需要设置softmax激活函数
        # loss function: cross entropy
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        # bind the layers
        y_pred = self.linear(x)
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            y_pred = self.activation(y_pred) # 不执行CrossEntropyLoss，但仍然需要执行激活函数
            return y_pred
    
# generate data: random 5D
# rule: the matrix index where the maximum number is located is the classification
def build_data():
    x = np.random.random(5)
    '''
    maxValue = x[0]
    index = 0
    # get the index where the max number is located
    for i in range(x.size):
        if x[i] > maxValue:
            index = i
            maxValue = x[i]
    '''
    return x, np.argmax(x)

def build_dataset(total_num):
    X = []
    Y = []
    for i in range(total_num):
        x, y = build_data()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# test for every epoch
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  
            if np.argmax(y_p) == int(y_t): # 与真实标签进行对比,概率最大的一个维度索引与实际分类做对比
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # model parameters
    epoch_num = 200
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.1

    # construct a module
    model = TorchModel(input_size)

    # chose a optimizer in torch
    # set learning rate
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # generate training samples
    train_x, train_y = build_dataset(train_sample)

    # print
    log = []
    # start training
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y) # calculate loss
            loss.backward() # calculate grad
            optim.step() # update weight
            optim.zero_grad() # zero grad
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.bin")
    # 画图
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度，训练时已经计算过了，预测时不必要再计算
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        print("输入：", input_vec)
        print("预测结果：", result)
    for vec, res in zip(input_vec, result):
        index = np.argmax(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, index, res[index]))  # 打印结果
        pass

if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                 [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                 [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                 [0.99349776,0.89416669,0.92579291,0.91567412,0.9358894]]
    predict("model.bin", test_vec)