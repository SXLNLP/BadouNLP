# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Step1:建立模型
class week2Model(nn.Module):
    def __init__(self,inputSize):
        super(week2Model,self).__init__()
        self.linear = nn.Linear(inputSize,5)    # 线性层输出五个类别的分数
        self.loss = nn.functional.cross_entropy # 损失函数选择交叉熵内置了激活函数 softmax
    # y有值返回loss证明在训练，y没值证明在预测返回预测值
    def forward(self,x,y=None):
        yyc = self.linear(x)
        if y is not None:
            y = y.squeeze().long()
            return self.loss(yyc,y)
        else:
            return yyc

# Step2:构造数据
# 生成单一样本方法
def build():
    x = np.random.random(5)
    maxIndex  = np.argmax(x)  # 最大值的索引
    return x, maxIndex

# 生成一批样本方法
def builds(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)
# Step3 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = builds(test_sample_num)
    # 统计各类别数量
    class_counts = [0] * 5
    for label in y:
        class_counts[int(label)] += 1
    print("本次预测集中各类别样本数量:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测类别
        for y_p, y_t in zip(predicted_classes, y.squeeze()):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# Step4:训练代码
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

    # 新建模型
    model = week2Model(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = builds(train_sample)
    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "week2Model.bin")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# Step5:训练后预测代码
def predict(model_path, input):
    input_size = 5
    model = week2Model(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input))  # 模型预测
    for vec, res in zip(input, result):
        predicted_class = torch.argmax(res).item()+1
        print(f"输入：{vec}, 预测类别：{predicted_class}, 各类别概率：{res.tolist()}")
# 主方法
if __name__ == "__main__":
    # 作业提交不提交权重文件 所以要先生成模型权重文件
    main()
    # 测试集
    test_vec = [
        [19.9, 11.1, 12.2, 13.3, 14.4],  # 第零位最大。为一类
        [21.1, 29.9, 22.2, 23.3, 24.4],  # 第一位最大。为二类
        [31.1, 32.2, 39.9, 33.3, 34.4],  # 第二位最大。为三类
        [41.1, 42.2, 43.3, 49.9, 44.4],  # 第三位最大。为四类
        [51.1, 52.2, 53.3, 54.4, 59.9]   # 第四位最大。为五类
    ]
    predict("week2Model.bin", test_vec)
