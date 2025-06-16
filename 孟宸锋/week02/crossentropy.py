import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 第一步，X:随机生成10000个五维数据，y:然后得到其中最大的那个数据
def build_dataset(num_samples):
    x = np.random.randint(1, 101, size=(num_samples, 5))
    y = np.argmax(x, axis=1)
    return torch.FloatTensor(x), torch.LongTensor(y)


# 第二步，用pytorch 建立一个模型
# 1. 定义模型
class TorchModel(torch.nn.Module):
    def __init__(self, input_size,):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(32, input_size)
        #self.cross_entropy = torch.nn.CrossEntropyLoss() # loss 函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, hidden_size)
        x = self.activation1(x)  # (batch_size, hidden_size) -> (batch_size, hidden_size)
        x = self.linear2(x)  # (batch_size, hidden_size) -> (batch_size, hidden_size)
        x = self.activation2(x)  # (batch_size, hidden_size) -> (batch_size, hidden_size)
        y_pred = self.linear3(x)  # (batch_size, hidden_size) -> (batch_size, input_size)

        return y_pred  # 输出预测结果


 
# 准确率评估函数
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    X, Y = build_dataset(1000)
    test_dataset = torch.utils.data.TensorDataset(X, Y)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_pred = model(inputs)
            predicted = torch.argmax(y_pred, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy



def main():
    num_epochs = 1000
    batch_size = 32
    learning_rate = 0.001
    input_size = 5
    train_sample_per_epoch = 500
    k_fold = 10
    #建立模型
    model = TorchModel(input_size)
    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()
    log = []

    # 1. 生成数据集
    X, Y = build_dataset(10000)

    # 2. 将数据集分为训练集和测试集
    train_size = int(0.8 * len(X))

    train_dataset = torch.utils.data.TensorDataset(X[:train_size], Y[:train_size])

    # 3. 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


    # 4. 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            acc = evaluate(model)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            log.append((acc, avg_loss))

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
    


if __name__ == "__main__":
    main()
