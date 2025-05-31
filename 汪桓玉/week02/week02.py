import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
import os

# 1⃣️【第二周作业】

# 改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

# 输入：5维向量
# 输出：五维--->取最大值下标 为类别11111111111111111
# 模型  线性->relu->线性->交叉熵

# 创建torch模型
class TorchModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(TorchModel,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1) # dim=1 表示在维度1上进行softmax
        self.loss = nn.functional.cross_entropy
    def forward(self,x,y=None):
        x = self.linear1(x)
        hidden1 = self.relu(x)
        y_pred = self.linear2(hidden1)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            y_pred = self.softmax(y_pred)
            print('y',y_pred)
            return y_pred

# 数据归一化函数
def normalize_data(data):
    # 针对每个样本分别归一化，保持相对大小关系
    data_min = np.min(data, axis=1, keepdims=True)
    data_max = np.max(data, axis=1, keepdims=True)
    # 避免除零错误
    data_range = np.maximum(data_max - data_min, 1e-10)
    normalized_data = (data - data_min) / data_range
    return normalized_data

def build_sanple():
    sample = np.random.uniform(0, 100000, 5)  # 修改为0-100范围
    max_index = np.argmax(sample)
    return sample, max_index
  

def create_train_sanple(train_sample):
    X = []
    Y = []
    for _ in range(train_sample):
        x,y = build_sanple()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    
    # 对数据进行归一化处理
    X_normalized = normalize_data(X)
    
    return torch.FloatTensor(X_normalized), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample = 100
    x,y = create_train_sanple(test_sample)
    with torch.no_grad():
        _,acc = calculate_result(model,x,y)
    return acc

def predict(model_src):
    input_size = 5  # 修改为正确的输入维度
    hidden_size = 10  # 添加隐藏层维度
    output_size = 5  # 添加输出维度

    eval_size = 10000 # 预测个数
    model = TorchModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_src, weights_only=True))
    x,y = create_train_sanple(eval_size)
    model.eval() # 测试模式
    with torch.no_grad(): # 不计算梯度
        print(f'总预测样本：{eval_size}')
        calculate_result(model,x,y)
          
        

def calculate_result(model,x,y):   
    correct = 0
    error = 0
    y_pred = model(x)
    y_pred = torch.argmax(y_pred,dim=1)
    for y_p,y_t in zip(y_pred,y):
        if(y_p == y_t): correct+=1
        else: error+=1
    print("总数：%d 正确预测个数：%d, 错误个数：%d 正确率：%f" % (correct+error, correct, error, correct / (correct + error)))  
    return correct, correct / (correct + error)

def main():
    #配置参数
    epoch_num = 20 #训练轮数
    batch_size = 20 #每此训练个数(每轮输入)
    train_sample = 10000 # 总训练样本(总输入)
    input_size = 5 # 输入维度
    learning_rate = 0.001 # 学习率
    hidden_size = 10 # 隐藏层维度
    output_size = 5 # 输出维度

    #建立模型
    model = TorchModel(input_size,hidden_size,output_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # 画图
    log = []
    # 创建训练集，正常任务是读取训练集

    for epoch in range(epoch_num):
        x_train,y_train = create_train_sanple(train_sample)
        model.train()
        watch_loss = []
        for batch_index in range(train_sample//batch_size):
            x = x_train[batch_index * batch_size:(batch_index + 1) * batch_size] # 20 * 5
            y = y_train[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x,y) # 计算loss
            loss.backward() # 计算梯度
            optimizer.step() # 更新权重
            optimizer.zero_grad() # 梯度归0
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建保存路径
    save_path = os.path.join(current_dir, 'week02model.bin')
    # 保存模型``
    torch.save(model.state_dict(),save_path)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
if __name__ == '__main__':
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建保存路径
    path = os.path.join(current_dir, 'week02model.bin')
    
    # 检查模型文件是否存在
    if not os.path.exists(path):
        print("模型文件不存在，开始训练...")
        main()
    else:
        print("找到模型文件，开始预测...")
    
    predict(path)