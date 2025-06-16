import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# 参数设置
vocab_size = 27  # a-z加上特殊字符
max_length = 20  # 字符串最大长度
samples = 10000  # 样本数量
batch_size = 64  # 批次大小
embedding_dim = 16  # 嵌入维度
hidden_dim = 32  # 隐藏层维度
epochs = 20  # 训练轮次

# 序列填充函数（全局定义）
def collate_fn(batch):
    """
    对批次数据进行填充和截断处理

    参数:
        batch: 包含多个样本的列表，每个样本为(序列张量, 标签)元组

    返回:
        tuple: 填充后的序列张量和堆叠后的标签张量
    """
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    if padded.size(1) > max_length:
        padded = padded[:, :max_length]
    return padded, torch.stack(labels)

# 生成数据
def generate_data(samples, max_length):
    """
    生成包含随机字符且包含特定标记字符'a'的序列数据集

    参数:
        samples: 需要生成的样本总数
        max_length: 序列的最大允许长度

    返回:
        tuple: 包含特征序列(X)和目标位置(y)的元组
    """
    X = []
    y = []

    for _ in range(samples):
        length = np.random.randint(1, max_length + 1)
        a_position = np.random.randint(0, length)

        chars = []
        for i in range(length):
            if i == a_position:
                chars.append('a')
            else:
                char_code = np.random.randint(98, 123)
                chars.append(chr(char_code))

        seq = [ord(c) - 96 for c in chars]  # 将字符转换为数值索引
        X.append(seq)
        y.append(a_position)

    return X, y

# 创建数据集类
class TextDataset(Dataset):
    """
    自定义数据集类，用于加载序列数据和对应标签

    属性:
        sequences: 存储特征序列的列表
        labels: 存储目标位置的张量
        max_length: 序列最大长度限制
    """
    def __init__(self, sequences, labels, max_length):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), self.labels[idx]

# 定义RNN模型
class RNNClassifier(nn.Module):
    """
    基于RNN的分类模型，用于预测特定字符首次出现的位置

    属性:
        embedding: 嵌入层，将离散字符索引映射为连续向量
        rnn: RNN层，处理序列数据
        fc: 全连接层，输出分类结果
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播过程

        参数:
            x: 输入序列张量，形状为(batch_size, sequence_length)

        返回:
            tensor: 输出预测结果，形状为(batch_size, output_dim)
        """
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

# 封装训练逻辑
def train_epoch(model, loader, optimizer, criterion, device):
    """
    执行单个训练周期的完整流程

    参数:
        model: 待训练的神经网络模型
        loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备(CPU/GPU)

    返回:
        tuple: 平均损失和准确率百分比
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total

# 封装验证逻辑
def evaluate(model, loader, criterion, device):
    """
    评估模型在验证/测试集上的性能

    参数:
        model: 待评估的神经网络模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备(CPU/GPU)

    返回:
        tuple: 平均损失和准确率百分比
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total

# 预测示例
def predict_example(model, sample, true_label, device):
    """
    展示模型对单个样本的预测结果

    参数:
        model: 训练好的神经网络模型
        sample: 单个特征序列样本
        true_label: 真实的目标位置标签
        device: 计算设备(CPU/GPU)
    """
    model.eval()
    # 加载最优模型权重
    model.load_state_dict(torch.load(best_model_path))
    print("🔁 已加载最优模型权重")
    with torch.no_grad():
        # ✅ 创建张量后立即移动到指定设备
        sample_tensor = torch.tensor(sample, dtype=torch.long).to(device)

        # 填充/截断逻辑
        if len(sample_tensor) > max_length:
            sample_tensor = sample_tensor[:max_length]
        else:
            padding = torch.zeros(max_length - len(sample_tensor), dtype=torch.long).to(device)
            sample_tensor = torch.cat((sample_tensor, padding))

        # 添加batch维度
        sample_tensor = sample_tensor.unsqueeze(0)

        # 模型预测
        output = model(sample_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()

        # 转换回字符串表示
        chars = []
        for idx in sample_tensor.squeeze(0).cpu().numpy()[:len(sample)]:
            if idx == 0:
                chars.append('-')
            else:
                chars.append(chr(idx + 96))
        string = ''.join(chars)

        print(f"示例字符串: {string}")
        print(f"真实首次出现'a'的位置: {true_label}")
        print(f"预测首次出现'a'的位置: {predicted_label}")
        print(f"预测概率分布: {probs.cpu().numpy().round(3)}")

if __name__ == '__main__':
    # 设置GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    best_val_acc = 0.0
    best_model_path = 'best_model.pth'

    # 生成数据
    X, y = generate_data(samples, max_length)

    # 分割数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # 创建数据加载器
    train_dataset = TextDataset(X_train, y_train, max_length)
    val_dataset = TextDataset(X_val, y_val, max_length)
    test_dataset = TextDataset(X_test, y_test, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

    # 初始化模型
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, max_length).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # for epoch in range(epochs):
    #     train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    #     val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    #
    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)
    #     train_accs.append(train_acc)
    #     val_accs.append(val_acc)
    #
    #     print(
    #         f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    #     # 保存最优模型
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), best_model_path)
    #         print(f"✅ 保存最优模型，验证准确率: {val_acc:.2f}%")
    # # 评估模型
    # test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # print(f'测试集结果: Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
    #
    # # 可视化训练过程
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='训练损失')
    # plt.plot(val_losses, label='验证损失')
    # plt.title('训练和验证损失')
    # plt.xlabel('轮次')
    # plt.ylabel('损失')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(train_accs, label='训练准确率')
    # plt.plot(val_accs, label='验证准确率')
    # plt.title('训练和验证准确率')
    # plt.xlabel('轮次')
    # plt.ylabel('准确率 (%)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()




    # 显示几个预测示例
    for i in range(3):
        print(f"\n=== 示例 {i + 1} ===")
        predict_example(model, X_test[i], y_test[i], device)
