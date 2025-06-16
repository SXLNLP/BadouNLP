import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，五维随机向量最大的数字在哪维就属于哪一类

"""

## 定义tensor框架
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5)  # 5分类任务，输出是每个类别的概率
        self.loss = nn.functional.cross_entropy  # 多分类任务采用交叉熵作损失函数，会自动对线性层后的返回值作softmax操作

    # 输入x，返回预测值，否则返回loss
    def forward(self,x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            y = y.long()
            return self.loss(y_pred, y)
        else:
            return nn.functional.softmax(y_pred, dim=1) # 返回各个类别的概率值


# 生成y
def get_y(x):
    """

    :param x: 5为数组
    :return: x中最大的索引+1
    """
    max_x = max(x)
    if max_x == x[0]:
        return 0
    elif max_x == x[1]:
        return 1
    elif max_x == x[2]:
        return 2
    elif max_x == x[3]:
        return 3
    else:
        return 4

# 生成样本
def build_sample(sample_num):
    """

    :param sample_num: 需要的随机样本数量
    :return: x_list张量，里面是5维numpy数组；y_list张量：对应x的组别[2,1,3..
    """
    x_list = []
    y_list = []
    for i in range(sample_num):
        x = np.random.random(5)
        y = get_y(x)
        x_list.append(x)
        # 注意这里的[y]
        y_list.append(y)
    return torch.FloatTensor(x_list), torch.LongTensor(y_list)


#print(build_sample(3))



# 定义评估函数
def evaluate(model):
    model.eval() # 声明进入评估模式
    # 随机生成测试集
    test_num = 100
    X, Y = build_sample(test_num)
    # 查看测试集数据
    y1_count = (Y == 0).sum().item()
    y2_count = (Y == 1).sum().item()
    y3_count = (Y == 2).sum().item()
    y4_count = (Y == 3).sum().item()
    y5_count = (Y == 4).sum().item()
    print("本次测试数据 %d 条，5种类型分布条数为 %d,%d,%d,%d,%d" % (
        test_num, y1_count, y2_count, y3_count, y4_count, y5_count
    ))
    with torch.no_grad():
        Y_pred = model(X)
        # 获取每行最大值和索引 返回值也是张量
        max_values,max_indexs = torch.max(Y_pred, dim=1) # 是2D张量（形状 [batch_size, num_classes]），才需要用 dim=1 取每行的最大值

        condition = (Y == max_indexs)
        correct_count = condition.sum().item()

        print("预测正确个数 %d；正确率 %f" % (correct_count, round(correct_count/test_num,2)))
        return round(correct_count/test_num,2)

## 训练
def main():
    epoch_num = 50 # 训练轮数
    sample_num = 5000 # 生成的随机样本个数
    input_size = 5 # 输入特征维度
    batch_size = 20

    learning_rate = 0.001 #学习率
    # 实例化
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 生成随机化样本
    X, Y = build_sample(sample_num)
    loss_list = []
    log = []
    for epoch in range(epoch_num):
        model.train() # 显式进入训练模式（推荐放在 epoch 循环内）
        for batch_index in range(sample_num // batch_size):
            x_batch = X[batch_index * batch_size:(batch_index+1) * batch_size]
            y_batch = Y[batch_index * batch_size:(batch_index+1) * batch_size]
            loss = model(x_batch, y_batch) # 计算损失值，返回值是一维张量，需要调用item()取元素
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 权重归0
            loss_list.append(loss.item())

        print("----第 %d 轮训练---,平均loss: %f" % (epoch, np.mean(loss_list)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(loss_list))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 预测
def predict(model_path, input_inv):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path)) # 加载训练好的权重

    model.eval() # 测试模式
    with torch.no_grad():
        result = model(torch.FloatTensor(input_inv))
    for vec,res in zip(input_inv, result):
        #print("res",res)
        max_value, max_index = torch.max(res, dim=0) # 1d张量，指定dim=0或者不加dim也行
        print("输入 %s, 预测类别 %d,  预测概率 %f" % (vec, max_index.item(), max_value.item()))



if __name__ == '__main__':

    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
