import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，第几个数最大，就是哪一类
技术方案：
    1、既然是多分类，激活函数就要使用能够实现多分类的Softmax激活函数，而不能使用二分类的Sigmoid激活函数
    2、由于激活函数得到的结果不是一个标量值，而是一个矩阵，所以使用交叉熵损失函数，而不使用均方差损失函数
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5) #定义1个线性层
        """
            问题1：为啥这里不需要再次指定soft激活函数，是因为Linear层已经自带了softmax吗？？？？？？但是试了下，加上self.activation = torch.softmax这一行，也没任何影响。
        """
        self.activation = torch.softmax  # Softmax激活函数
        self.loss = nn.functional.cross_entropy # loss函数采用交叉熵

    def forward(self, x, y=None):
        print("x：\n", x)
        """
            问题2：同理，这里的预测值为啥直接使用线性层生成的值，而不需要经过激活函数转换？？？？？？
        """
        y_pred = self.linear(x)
        #y_pred = self.activation(x)
        print("y_pred：\n", y_pred)
        print("y：\n", y)

        # x = self.linear(x)
        # y_pred = self.activation(x, dim = 0)

        # print("x：\n", x)
        # print("y_pred：\n", y_pred)
        # print("y：\n", y)
        # print(" self.loss(y_pred, y)：\n", self.loss(y_pred, y))

        if y is not None:
            #print(" self.loss(y_pred, y)：\n", self.loss(y_pred, y))
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

#生成一个元组，第一个元素是一个5维的数组，第二个元素也是一个5维度的数组，第1个数组中哪个下表的元素最大，第二个元素5维数组中的第几个数字就是1
def build_sample():
    x = np.random.random(5) #生成一个5个数字的列表
    max_index = np.argmax(x)
    return x, max_index

# 随机生成一批样本
# 正负样本均匀生成
# 返回数据是一个元组，第一个元素是一个total_sample_num * 5的张量，第二个元素是total_sample_num * 1的张量
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        #Y.append([y])
        """
            问题3：这里的y得是1维张量，不能用2维张量。否则报错 RuntimeError: 0D or 1D target tensor expected, multi-target not supported
            但是之前的二分类找数字规律任务用的2维张量就可以
        """
        Y.append(y)
    # return torch.FloatTensor(X), torch.FloatTensor(Y)
    """
    这里的y的类型是Long，如果用Float就报错RuntimeError: expected scalar type Long but found Float。
    但是之前的二分类找数字规律任务也，用torch.FloatTensor(Y)也没事，为啥？？？？
    """
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    """"""""""""""" model.eval()的作用是干啥的？？？？？？？？？？？  """""""""""""""
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # if np.argmax(y_p) == np.argmax(y_t):
            if torch.argmax(y_p) == int(y_t):
                correct += 1  # 样本预测正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    """"""""""""""" torch.optim.Adam是用来干啥的。  """""""""""""""
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        """"""""""""""" mmodel.train()用来将模块设置为训练模式。  """""""""""""""
        model.train()
        watch_loss = []
        """"""""""""""" train_sample // batch_size，执行的是向下取整地板除，可能会有部分数据未参与到训练  """""""""""""""
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            print("=========\n第%d轮、第%d批的loss:%f" % (epoch + 1, batch_index +1, loss))

            """"""""""""""" loss.backward()反向传播，根据当前的损失值计算损失函数相对于模型参数的梯度  """""""""""""""
            loss.backward()  # 计算梯度

            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

            """"""""""""""" model.parameters()查看模型参数，含权重矩阵、偏置等信息。含两个张量，第1个张量是权重矩阵（即w），第2个是偏置(即b)  """""""""""""""
            #print("=========\n第%d轮、第%d批的模型参数\n" % (epoch + 1, batch_index +1), list(model.parameters()))
            """
                [
                Parameter containing: tensor([[0.3738, -0.3008, 0.2924, 0.2862, -0.4181]], requires_grad=True), 
                Parameter containing: tensor([-0.1995], requires_grad=True)
                ]
            """

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

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print("加载的模型权重：\n", model.state_dict())

    """
    model.eval()的作用：用于将模型切换到评估模式的方法，主要作用包括禁用 Dropout 层和使用全局统计量的归一化层。
        ‌归一化层的行为调整：在评估模式下，BatchNorm（批量归一化）等层会使用训练阶段累积的全局均值和方差进行数据标准化，而非当前批次的统计量。这保证了推理时的稳定性和一致性。
        禁用Dropout层：在训练时会随机丢弃神经元以防止过拟合，但评估模式下需固定网络结构以输出稳定结果。‌ model.eval() 会直接关闭 Dropout 的随机性，确保所有神经元均参与计算。
    使用场景：
        推理阶段‌：在测试集验证、模型部署或生成预测结果时，必须调用 model.eval()，避免因 Dropout 或动态归一化导致结果波动。
        与 torch.no_grad() 配合使用‌：通常与 with torch.no_grad(): 结合，进一步减少显存占用并加速计算（停止梯度跟踪）。
    """
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果

# main()
if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894],
    #             [0.39349776,0.59416669,0.92579291,0.41567412,0.1358894],
    #             [0.49349776,0.59416669,0.92579291,0.41567412,0.2358894]]
    # predict("model.bin", test_vec)

# t = build_dataset(2)
# print("t：\n",t)

# t = build_sample()
# print("t：\n",t)