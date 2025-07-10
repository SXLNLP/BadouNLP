import torch
import torch.nn as nn
import numpy as np

"""
****************注意：示例代码中每一行代码都要搞懂****************
"""

#定义一个2层的神经网络模型，每层都是一个线性层
class TorchModel(nn.Module):  #要继承nn.Module类
    def __init__(self, input_size, hidden_size1, hidden_size2):
        #调用父类的构造函数
        super(TorchModel, self).__init__()
        """
        nn.Linear参数说明：
            in_features - 输入特征的数量
            out_features - 输出特征的数量
        """
        self.layer1 = nn.Linear(input_size, hidden_size1)
        # self.sig = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        # self.relu = nn.ReLU()

    #预测函数
    def forward(self, x):
        x= self.layer1(x)
        # x = self.sig(x)
        y_pred = self.layer2(x)
        # y_pred = self.relu(y_pred)
        return y_pred

#自定义模型
class DiyModel:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        #x: 2 * 3, w1.T: 3 * 5   = 2*5  b1: 1 * 5
        print("###########################")
        # print("np.dot(x, self.w1.T)：\n", np.dot(x, self.w1.T))
        # print("self.b1：\n", self.b1)
        # print("np.dot(x, self.w1.T) +  self.b1 ：\n", np.dot(x, self.w1.T) + self.b1)
        print("###########################")
        hidden = np.dot(x, self.w1.T) + self.b1 #1*5  2*5
        y_pred = np.dot(hidden, self.w2.T) + self.b2 #1*2   2*5 dot  5*2  >  2*2     2*2+1*2 > 2*2
        return y_pred

#随便准备一个网络输入
x = np.array([[1.2, 2.3, 3.8], [4.5, 5.2, 6.4]])
# x = np.array([[1.2, 2.3, 3.8]])

#建立torch模型
"""
    这里有一个问题??????????????????????????????????????：
    1、TorchModel类的三个参数input_size、hidden_size1、hidden_size2试了下可以随便填，貌似跟样本数据x的形状、维度没有任何关系吗？
"""
torch_model = TorchModel(3, 5, 2)
"""
torch_model.state_dict 重要属性
    weight - 可学习的权重参数（形状为[out_features, in_features]）
    bias - 可学习的偏置参数（形状为[out_features]）
"""
# print(torch_model.state_dict())
print("--------------------------------------------")
#打印模型权重，权重为随机初始化。将张量转换为numpy列表
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()
print("torch w1 权重：\n", torch_model_w1)
print("torch b1 权重：\n", torch_model_b1)
print("--------------------------------------------")
print("torch w2 权重：\n", torch_model_w2)
print("torch b2 权重：\n", torch_model_b2)
print("--------------------------------------------")
#使用torch模型做预测
torch_x = torch.FloatTensor(x)
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：\n", y_pred)


# #把torch模型权重拿过来自己实现计算过程
diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
# #用自己的模型来预测
y_pred_diy = diy_model.forward(np.array(x))
print("diy模型预测结果：", y_pred_diy)




