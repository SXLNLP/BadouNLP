# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
1. 改用交叉熵实现一个多分类任务
2. 规律：x是一个5维向量，五维随机向量最大的数字在哪维就属于哪一类。
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，輸出defined的類別數量
        self.loss = nn.CrossEntropyLoss() # 交叉熵损失

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 使用softmax函数将预测值转换为概率


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，五维随机向量最大的数字在哪维就属于哪一类
def build_sample() -> tuple:
    x = np.random.random(5)
    # 找出最大值所在的維度（索引）
    max_index = np.argmax(x)
    # 返回向量和最大值所在的維度（作為類別標籤）
    return x, max_index


# 随机生成一批样本
def build_dataset(total_sample_num) -> tuple:
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample() 
        X.append(x)  
        Y.append(y)
    X = torch.FloatTensor(np.array(X))  # 形狀為 [total_sample_num, 5]
    Y = torch.LongTensor(np.array(Y))   # 形狀為 [total_sample_num]
    return X, Y

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model) -> float:
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 統計每個類別的樣本數量
    class_counts = torch.zeros(5) # 初始化一個5維的張量，用於計數
    for label in y:
        class_counts[label] += 1 
    print("測試集中各類別樣本數量：")
    for i in range(5):
        print(f"類別 {i}: {int(class_counts[i])} 個樣本")
    
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型預測
        # 獲取預測的類別（取最大概率的類別）
        _, predicted = torch.max(y_pred, 1)
        # 計算正確預測的數量
        correct = (predicted == y).sum().item() 
    
    accuracy = correct / test_sample_num
    print(f"正確預測個數：{correct}, 正確率：{accuracy:.4f}")
    return accuracy 


def main():
    print("開始訓練...")
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数
    learning_rate = 0.001  # 学习率
    print(f"參數設置完成：epoch_num={epoch_num}, batch_size={batch_size}, train_sample={train_sample}")
    
    # 建立模型
    print("正在建立模型...")
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    print("正在生成訓練數據...")
    train_x, train_y = build_dataset(train_sample)
    print(f"訓練數據生成完成，形狀：X={train_x.shape}, Y={train_y.shape}")
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零 - 梯度不归零的话，会累加起来，迭代轮数多了容易梯度爆炸
            watch_loss.append(loss.item())

        # 計算平均loss
        avg_loss = np.mean(watch_loss)    
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, avg_loss])
        print("當前的準確率：%f" % acc)
    # 保存模型
    torch.save(model.state_dict(), "model-cross-entropy.bin")
    # 打印log
    print(log)
    # 畫圖
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1) # 畫第一個圖
    plt.title("Accuracy Curve")
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画acc曲线
    plt.xlabel("輪數")
    plt.ylabel("準確率")
    plt.legend()

    plt.subplot(1, 2, 2) # 畫第二個圖
    plt.title("Loss Curve")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲線
    plt.xlabel("輪數")
    plt.ylabel("損失")
    plt.legend()

    plt.tight_layout() # 調整子圖之間的間距
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path: str, input_vec: list) -> torch.Tensor:

    input_size = 5
    model = TorchModel(input_size, num_classes=5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

    for vec, probs in zip(input_vec, result):
        pred_class = torch.argmax(probs).item()
        pred_prob = probs[pred_class].item()
        print(f"輸入：{vec}, 預測類別：{pred_class}, 預測概率：{pred_prob:.4f}")
        print(f"預測結果：{pred_class}")
        for i, prob in enumerate(probs):
            print(f"類別 {i} 的概率：{prob:.4f}")


if __name__ == "__main__":
    main()
    
    # 複雜的預測樣本
    test_input = [
        [0.999, 0.998, 0.997, 0.996, 0.995],  
        [0.001, 0.002, 0.003, 0.004, 0.005],
        [0.9999, 0.0001, 0.0001, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0001, 0.0001, 0.9999],
        [0.9, 0.1, 0.8, 0.2, 0.7],
        [0.1, 0.9, 0.2, 0.8, 0.3],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
    ]
    # 預測樣本
    predict("model-cross-entropy.bin", test_input)
