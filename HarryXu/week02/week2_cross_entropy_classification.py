"""
Week2 作业：
用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层直接映射到5个类别
        self.loss = nn.CrossEntropyLoss()  # 多分类使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测类别
    def forward(self, x, y=None):
        logits = self.linear(x)
        if y is not None:
            return self.loss(logits, y.long())  # 计算交叉熵Loss
        else:
            return torch.argmax(logits, dim=1)  # 返回预测类别

# 生成单样本：随机5维向量，标签为最大值所在维度
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x) 
    return x, label

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print(f"测试集中各标签数量：{np.bincount(y.numpy())}")
    
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 直接获取预测类别
        correct = (y_pred == y).sum().item()
    
    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy

def main():
    # 超参数
    epoch_num = 20  
    batch_size = 20  
    train_sample = 5000  
    input_size = 5  
    learning_rate = 0.001  
    
    # 建立模型
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 生成训练数据
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        # 手动批处理（简化版）
        for i in range(0, train_sample, batch_size):
            x = train_x[i:i+batch_size]
            y = train_y[i:i+batch_size]
            loss = model(x, y)  # 计算损失（交叉熵）
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        
        # 输出loss
        avg_loss = np.mean(watch_loss)
        acc = evaluate(model)
        log.append([acc, avg_loss])
        print(f"第{epoch+1}轮 | 平均Loss: {avg_loss:.4f} | 验证准确率: {acc:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    print(log)
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# 预测函数
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        x = torch.FloatTensor(input_vec)
        y_pred = model(x)  # 直接获取预测类别
        
    for vec, label in zip(input_vec, y_pred):
        print(f"输入：{vec}, 预测类别：{label.item()}")

if __name__ == "__main__":
    main()
    
    # test_vec = [
    #     [0.1, 0.3, 0.5, 0.2, 0.4],  # 最大值在索引2
    #     [0.8, 0.2, 0.1, 0.3, 0.4],  # 最大值在索引0
    #     [0.2, 0.5, 0.3, 0.9, 0.1],  # 最大值在索引3
    # ]
    # predict("model.bin", test_vec)
