#week02-zy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义多分类模型（线性层+Softmax）
class MultiClassModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(MultiClassModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)  # 线性层：输入5维，输出5类
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失（内置Softmax）

    def forward(self, x, y=None):
        logits = self.linear(x)  # 输出未激活的logits
        if y is not None:
            return self.loss(logits, y)  # 训练时返回损失
        else:
            return torch.softmax(logits, dim=1)  # 推理时返回概率分布

# 生成单个样本：5维随机向量，标签为最大值所在维度
def build_sample():
    x = np.random.random(5)  # 生成5个[0,1)之间的随机数
    max_index = np.argmax(x)  # 获取最大值索引（类别标签）
    return x, max_index

# 生成批量数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 标签为整数（0-4）
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 评估模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        logits = model.linear(x)  # 直接获取logits
        pred = torch.argmax(logits, dim=1)  # 预测类别（取logits最大值索引）
    accuracy = (pred == y).sum().item() / test_sample_num
    print(f"测试集准确率：{accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    input_size = 5       # 输入维度
    class_num = 5        # 类别数（5类）
    epoch_num = 50       # 训练轮数
    batch_size = 100     # 批量大小
    train_sample = 10000 # 训练样本总数
    learning_rate = 0.01 # 学习率

    # 初始化模型、优化器
    model = MultiClassModel(input_size, class_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_list = []
    train_acc_list = []

    # 生成训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch_idx in range(0, train_sample, batch_size):
            # 提取批量数据
            x_batch = train_x[batch_idx:batch_idx+batch_size]
            y_batch = train_y[batch_idx:batch_idx+batch_size]
            
            # 计算损失并反向传播
            loss = model(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 记录训练损失和准确率
        train_loss = total_loss / (train_sample // batch_size)
        train_acc = evaluate(model)  # 评估函数已包含test模式切换
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        print(f"Epoch {epoch+1}/{epoch_num}, 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.pth")

    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch_num+1), train_loss_list, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch_num+1), train_acc_list, label="Accuracy", color="r")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()
    plt.show()

# 预测函数
def predict(model_path, input_vecs):
    model = MultiClassModel(input_size=5, class_num=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(input_vecs)
        probs = model(x)  # 获取概率分布
        preds = torch.argmax(probs, dim=1)  # 预测类别
    for vec, pred, prob in zip(input_vecs, preds, probs):
        print(f"输入向量: {vec}")
        print(f"预测类别: {pred.item()}, 概率分布: {prob.numpy()}")
        print("-"*50)

if __name__ == "__main__":
    main()
    
    # 测试预测（示例）
    test_vectors = [
        np.random.random(5),  # 随机向量1
        np.random.random(5),  # 随机向量2
        [0.1, 0.3, 0.9, 0.2, 0.4],  # 最大值在第2维
        [0.8, 0.2, 0.3, 0.1, 0.4],  # 最大值在第0维
        [0.1, 0.9, 0.3, 0.2, 0.5]   # 最大值在第1维
    ]
    predict("multiclass_model.pth", test_vectors)
