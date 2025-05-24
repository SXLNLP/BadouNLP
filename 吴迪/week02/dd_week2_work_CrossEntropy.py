import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

""" 
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""
#定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5) #定义一个线性层，输出5个类别
        self.loss = nn.CrossEntropyLoss() # 使用torch计算交叉熵（自动含Softmax）
    
    #输入真实值，返回 loss 计算损失,无真实标签返回预测值
    def forward(self,x,y = None):
        x = self.linear(x) #输入 x：原始logits（未经过softmax）
        if y is not None: #目标 y：类别索引（不是one-hot）
            y = y.long().view(-1) #交叉熵输出损失,使用 y = y.squeeze() 或 y = y.view(-1) 调整确保y是1D张量
            return self.loss(x,y) 
        else:
            return torch.softmax(x,dim=1) #预测时输出的概率分布

#生成一个有规律五维样本，最大索引作为标签
def build_sample():
    x = np.random.random(5)  # 随机5维向量
    y = np.argmax(x) #最大索引作为类别
    return x,y

#生成样本数据集
def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for _ in range(total_sample_num): #当循环变量不会被使用时（仅需重复操作），用 _ 表示“忽略这个变量”：
        x_data,y_data = build_sample()
        X.append(x_data)
        Y.append(y_data)
    return torch.FloatTensor(np.array(X)),torch.LongTensor(np.array(Y)) #先转bumpy，FloatTensor(X)浮点数张量存储特征，整数张量存储类别标签,直接对Python列表使用 torch.FloatTensor() 效率低

#评估函数，测试每轮模型准确率
def evaluate(model):
    model.eval() #关闭上述层的训练特性，切换到评估模式
    total_sample_num = 100 #假设有100个测试样本
    x,y = build_dataset(total_sample_num)
    
    with torch.no_grad(): #不计算梯度
        y_pred = model(x) #forwad 模型，预测y值，输出概率分布
        pred_classes = torch.argmax(y_pred,dim=1) #预测概率最大的类别，dim=1 每行取最大值的索引
        #逐元素比较预测类别和真实类别
        correct = (pred_classes == y).sum().item()  #.sum()统计出正确的数量，item()python中将张量转位Python数字
    acc = correct / total_sample_num       #统计正确率
    print(f"正确个数为:{correct},正确率为:{acc:.4f}")
    return acc

#训练过程    
def main():
    #配置参数
    epoch_num = 20 #训练总轮数
    batch_size = 20 #每批数据量
    train_sample = 5000 #训练样本总数
    input_size = 5 #特征5维向量
    learning_rate = 0.01 #学习率，控制参数更新的步长
    
    #初始化模型和优化器
    model = TorchModel(input_size) #建立模型
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate) #Adam优化器
    log=[]#记录训练日志
    
    #数据准备，创建训练集
    train_x,traim_y = build_dataset(train_sample) #train_x 是5000个5维向量，train_y 每个向量对应的最大数字类别所属的标签
    
    #外循环训练，epoch遍历所有数据一次，称为1轮
    for epoch in range(epoch_num):#共训练20轮
        model.train() #设置为训练模式 
        watch_loss = [] #记录当前轮数的损失值       
        #内循环
        for batch_idx in range(train_sample // batch_size):
            #获取当前批次的数据
            x = train_x[batch_idx*batch_size : (batch_idx+1)*batch_size] 
            y = traim_y[batch_idx*batch_size : (batch_idx+1)*batch_size]
            
            #前向传播计算损失
            loss = model(x,y) #调用forward(x,y)计算损失
            #反向传导计算损失
            loss.backward() #自动计算新梯度
            #更新参数
            optim.step() #根据梯度更新模型参数
            optim.zero_grad()#清空上一轮的梯度
            #记录损失
            watch_loss.append(loss.item()) #保存当前批次的损失值
        avg_loss = np.mean(watch_loss) #计算当前轮次的平均损失
        print(f"第{epoch+1}轮,平均loss:{avg_loss:.4f}")
        
        acc = evaluate(model) #在测试集评估准确率
        log.append([acc,avg_loss]) #记录准确率和损失值
    
    #绘制准确率和损失曲线
    plt.plot(range(len(log)),[l[0] for l in log],label="acc")   #准确率曲线
    plt.plot(range(len(log)),[l[1] for l in log],label="avg_loss") #损失曲线
    plt.legend() #显示图例
    plt.show() #显示图像
    
    #保存模型参数（state_dict)只保存参数, 不需要 return，或确保它在最后
    torch.save(model.state_dict(),"model.pth")     
    
#使用训练好的模型，做预测,model_path为保存的模型文件的路径，input_vec是药预测的输入数据的列表
def predict(model_path,input_vec):
    model=TorchModel(5) #新建模型实例（输入维度5），与原模型相同
    model.load_state_dict(torch.load(model_path)) #加载保存的数据模型
    model.eval() #设置为评估模式，确保预测结果准确稳定一致
    with torch.no_grad(): #禁用梯度，节省内存，加速计算
        prod = model(torch.FloatTensor(input_vec)) #输入转为张量，并预测
        pred_classes = torch.argmax(prod,dim=1) #取概率最大的类别，输出预测类别的索引
    for vec,cls,p in zip(input_vec,pred_classes,prod):
        print(f"输入：{vec},预测类别：{cls.item()},各类概率：{p.numpy().round(4)}")

#结果
if __name__ == "__main__":
    main()
    test_vec = [
        [0.1, 0.9, 0.2, 0.3, 0.4],   # 最大值在1（第2类）
        [0.8, 0.1, 0.6, 0.4, 0.3],   # 最大值在0（第1类）
        [0.2, 0.3, 0.4, 0.5, 0.1]    # 最大值在3（第4类）
    ]
    predict("model.pth",test_vec)
