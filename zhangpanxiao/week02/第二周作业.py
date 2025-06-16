import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score



# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 1.定义数据生成函数
def generate_data(batch_size=32):
    """生成随机五维向量，标签是最大值的索引"""
    # 生成随机数据 (batch_size x 5)
    data = np.random.rand(batch_size, 5)
    # 计算每行最大值的索引 (0-4)
    label = np.argmax(data, axis=1)
    return torch.FloatTensor(data), torch.LongTensor(label)


# 2.定义简单模型
class FiveClassClassifier(nn.Module):
    def __init__(self):
        super(FiveClassClassifier, self).__init__()

        # 两层网络
        self.fc1 = nn.Linear(5, 16)  # 输入5维，输出16维
        self.fc2 = nn.Linear(16, 5)  # 输入16维，输出5维（对应5个类别）

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)  # 注：不需要softmax，因为CrossEntropyLoss会自动处理
        return x


# 3.定义训练函数
def train_model(model, epochs=100, batch_size=32, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # 生成训练数据
        inputs, labels = generate_data(batch_size)
        # 梯度清零
        optimizer.zero_grad()
        # 输出结果
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()

        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 4.定义预测函数 使用训练好的模型进行预测
def predict(model, input_data):

    # 确保模型在评估模式
    model.eval()

    # 禁用梯度计算
    with torch.no_grad():

        # 转换输入数据为PyTorch张量
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data )

        outputs = model(input_data)

        # 获取最大值所在维度的索引
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


# 5.主程序
if __name__ == '__main__':
    # 初始化模型
    model= FiveClassClassifier()

    # 训练模型
    print('开始训练...')
    train_model(model,epochs=100)
    print('训练完成!')

    #测试模型
    # 生成测试数据集
    test_data, test_labels = generate_data(5)
    print('\n测试数据:')
    for i, (data, label) in enumerate(zip(test_data, test_labels)):
        # 预测
        pred = predict(model, data.unsqueeze(0))  # 添加batch维度
        # 显示结果
        print(f"样本 {i + 1}:")
        print(f"  输入向量: {data.numpy().round(4)}")
        print(f"  真实类别: {label.item()} (最大值在第{label.item() + 1}维)")
        print(f"  预测类别: {pred} (最大值在第{pred + 1}维)")
        print("-" * 40)

    # 评估准确率
    # 生成1000个测试数据
    test_data, test_labels=  generate_data(1000)
    # 禁用梯度
    with  torch.no_grad():
        predictions = [predict(model, data.unsqueeze(0)) for data in test_data]
        accuracy = accuracy_score(test_labels, predictions)
        print(f"\n测试准确率: {accuracy * 100:.2f}%")

