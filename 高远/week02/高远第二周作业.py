# 第二周作业：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
import torch
import torch.nn as nn

# 设置随机种子确保结果可复现
torch.manual_seed(42)

# 使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()

# 生成5个样本，每个样本是5维向量，做5分类任务
# 这里使用随机生成的数据代替原代码中的示例数据
pred = torch.randn(5, 5)  # 5个样本，每个样本5维
target = torch.argmax(pred, dim=1)  # 标签为最大值所在维度

# 计算torch的交叉熵损失
loss = ce_loss(pred, target)
print(f"Torch交叉熵输出: {loss.item():.4f}")


# 手动实现softmax函数 - 使用PyTorch实现
def softmax_torch(logits):
    # 为了数值稳定性，减去最大值
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)


# 手动实现交叉熵 - 使用PyTorch实现
def cross_entropy_torch(logits, target):
    # 应用softmax函数
    probs = softmax_torch(logits)

    # 获取目标类别的概率
    batch_size = logits.shape[0]
    correct_probs = probs[range(batch_size), target]

    # 计算交叉熵
    return -torch.mean(torch.log(correct_probs + 1e-10))  # 添加小常数避免log(0)


# 计算手动实现的交叉熵损失
manual_loss = cross_entropy_torch(pred, target)
print(f"手动实现交叉熵输出: {manual_loss.item():.4f}")

# 验证两种实现的结果是否一致
assert torch.allclose(loss, manual_loss, atol=1e-5), "两种实现的结果不一致!"
print("验证通过: 两种实现的结果一致!")


# 创建一个简单的神经网络模型用于多分类任务
class SimpleClassifier(nn.Module):
    def __init__(self, input_size=5, num_classes=5):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


# 初始化模型和优化器
model = SimpleClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练模型
def train_model(model, optimizer, epochs=100):
    for epoch in range(epochs):
        # 生成一批新的随机训练数据
        inputs = torch.randn(32, 5)  # 32个样本，每个样本5维
        targets = torch.argmax(inputs, dim=1)  # 标签为最大值所在维度

        # 前向传播
        outputs = model(inputs)
        loss = ce_loss(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 测试模型
def test_model(model):
    # 生成测试数据
    inputs = torch.randn(10, 5)
    targets = torch.argmax(inputs, dim=1)

    # 模型预测
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    # 计算准确率
    accuracy = (predicted == targets).float().mean()
    print(f'测试准确率: {accuracy.item() * 100:.2f}%')


# 训练和测试模型
print("\n开始训练模型...")
train_model(model, optimizer)
print("\n测试模型性能:")
test_model(model)


# 输出结果
# Torch交叉熵输出: 0.8761
# 手动实现交叉熵输出: 0.8761
# 验证通过: 两种实现的结果一致!
#
# 开始训练模型...
# Epoch [10/100], Loss: 1.5958
# Epoch [20/100], Loss: 1.4613
# Epoch [30/100], Loss: 1.3137
# Epoch [40/100], Loss: 1.1142
# Epoch [50/100], Loss: 1.1197
# Epoch [60/100], Loss: 0.9459
# Epoch [70/100], Loss: 0.9281
# Epoch [80/100], Loss: 0.8847
# Epoch [90/100], Loss: 0.8440
# Epoch [100/100], Loss: 0.7213
#
# 测试模型性能:
# 测试准确率: 90.00%
