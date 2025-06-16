# 导入必要的库
import torch # PyTorch深度学习框架
import torch.nn as nn # 神经网络模块
import numpy as np # 数值计算库
import random # 随机数生成
import json # JSON数据处理（未使用）
import matplotlib.pyplot as plt # 数据可视化

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，从5个数中获取最大的数的下标值.将模型预测结果 与 实际下标进行比较.
"""

# 定义神经网络模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__() # 初始化父类
        # 定义单层线性网络：输入维度input_size，输出维度5
        self.linear = nn.Linear(input_size, 5)
        # 使用均方误差损失函数（适用于回归问题，这里用于二分类）
        self.loss = nn.CrossEntropyLoss()

    # 前向传播函数
    def forward(self, x, y=None):
        logits = self.linear(x)
        if y is not None:
            return self.loss(logits, y.squeeze().long())
        else:
            return logits

# 生成单个样本
def build_sample():
    x = np.random.random(5) # 生成5维随机向量
    max_index = np.argmax(x)
    return x,max_index

# 生成训练数据集
def build_dataset(total_sample_num):
    x_set = [] # 存储特征
    y_set = [] # 存储标签
    for i in range(total_sample_num):
        x, y = build_sample() # 生成单个样本
        x_set.append(x) # 添加特征
        y_set.append([y])
    x_set = np.array(x_set)
    y_set = np.array(y_set)
    return torch.FloatTensor(x_set), torch.LongTensor(y_set)

# 模型评估函数
def evaluate(model):
    model.eval() # 设置为评估模式
    test_sample_num = 100 # 测试样本数
    x, y = build_dataset(test_sample_num) # 生成测试集
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0 # 初始化计数器
    with torch.no_grad(): # 禁用梯度计算
        y_pred = model(x) # 模型预测
        predicted = torch.argmax(y_pred,dim=1)
        # 如果模型预测值 和 实际分类的标记一致 则正确值+1
        for y_p ,y_t in zip(predicted,y):
            if y_t == y_p:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
    # return accuracy

# 训练主函数
def main():
    # 超参数设置
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 批次大小
    train_sample = 5000 # 每轮训练样本总数
    input_size = 5 # 输入特征维度
    learning_rate = 0.001 # 学习率

    # 初始化模型
    model = TorchModel(input_size)
    # 选择Adam优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = [] # 记录训练日志

    # 生成训练数据集
    train_x, train_y = build_dataset(train_sample)

    # 开始训练循环
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample //batch_size):
            x = train_x[batch_index * batch_size:(batch_index +1 )* batch_size]
            y = train_y[batch_index * batch_size:(batch_index +1 )* batch_size]
            optim.zero_grad()
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        # 打印训练信息
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model) # 评估模型
        log.append([acc, float(np.mean(watch_loss))]) # 保存日志

    # 保存模型参数
    torch.save(model.state_dict(), "model1.bin")

    # 绘制训练曲线
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc") # 准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss") # 损失曲线
    plt.legend() # 显示图例
    plt.show() # 展示图表
    return

# 预测函数
def predict(model_path, input_vec):
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
        predicted = torch.argmax(result,dim=1)
    for vec,res in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, res))

# 程序入口
if __name__ == "__main__":
    main() # 运行训练流程
#  测试预测函数
    test_vec = [
        [0.21350445, 0.48720946, 0.42111406, 0.43671203 ,0.87120381],
        [0.76452188, 0.28403264, 0.87057438, 0.50835505 ,0.80547519],
        [0.36348148, 0.63338373, 0.96301205, 0.4962551  ,0.02512183],
        [0.66129961, 0.83804043, 0.13552003, 0.65757785 ,0.18408155],
        [0.17538116, 0.55823476, 0.85397191, 0.76512488 ,0.80683101],
    ]
    predict("model1.bin", test_vec)
#     for i in range(5):
#         print(np.random.random(5))
