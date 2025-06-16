import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),  # 输入1通道，输出6通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 输入6通道，输出16通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),  # 第一个全连接层
            nn.ReLU(),
            nn.Linear(120, 84),  # 第二个全连接层
            nn.ReLU(),
            nn.Linear(84, 10)  # 输出层，10个类别
        )

    def forward(self, x):
        x = self.conv1(x)  # 第一个卷积块
        x = self.conv2(x)  # 第二个卷积块
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc(x)  # 全连接层
        return x


# 加载MNIST数据集
def load_data():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
    ])

    # 加载训练集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # 6万条数据
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 加载测试集
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def train_model(train_loader, test_loader, device='cpu'):
    # 创建模型
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # 训练参数
    epochs = 10

    # 记录开始训练时间
    start = time.time()

    # 训练循环
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 计算训练集准确率和损失
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # 保存模型
    model_path = 'lenet5_minist.pth'
    torch.save(model.state_dict(), model_path)

    # 记录结束训练时间
    end = time.time()
    print(f"总训练时间: {end - start:.2f} 秒")


def load_mode_test(device='cpu'):
    # 验证集评估
    model_path = 'lenet5_minist.pth'
    model = LeNet5()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    val_loss = 0
    val_correct = 0
    val_total = 0
    val_losses = []
    val_accs = []
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            val_loss = val_loss / len(test_loader)
            val_acc = 100. * val_correct / val_total

            val_losses.append(val_loss)
            val_accs.append(val_acc)

    print('加载模型，验证集准确率： ', val_acc)
    return val_losses, val_accs


def plt_fig(losses, accs):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Testing Loss')
    plt.title('Testing Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accs, label='Testing Accuracy', color='orange')
    plt.title('v Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    train_loader, test_loader = load_data()
    # 模型训练
    train_model(train_loader, test_loader)
    # 模型测试
    val_losses, val_accs = load_mode_test()
    # 结果可视化展示
    plt_fig(losses=val_losses, accs=val_accs)

