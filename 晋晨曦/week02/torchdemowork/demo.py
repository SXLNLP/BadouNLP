import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y.squeeze().long())
        else:
            return torch.softmax(x, dim=1)

def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index

def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)

    # 统计每个类别的样本数
    class_counts = [0] * 5
    for label in y:
        class_counts[label.item()] += 1

    print("各类别样本分布:", class_counts)

    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        predicted = y_pred.argmax(dim=1)
        correct = (predicted == y).sum().item()

    acc = correct / test_sample_num
    print(f"正确预测数：{correct}, 正确率：{acc:.4f}")
    return acc

def main():

    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []


    train_x, train_y = build_dataset(train_sample)


    for epoch in range(epoch_num):
        model.train()
        epoch_loss = []


        indices = torch.randperm(train_sample)
        train_x = train_x[indices]
        train_y = train_y[indices]

        for batch_idx in range(0, train_sample, batch_size):
            end_idx = batch_idx + batch_size
            x = train_x[batch_idx:end_idx]
            y = train_y[batch_idx:end_idx]

            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        acc = evaluate(model)
        log.append([acc, avg_loss])
        print(f"Epoch {epoch+1}/{epoch_num} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot([l[0] for l in log], label="Accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot([l[1] for l in log], label="Loss", color='orange')
    plt.legend()
    plt.show()

def predict(model_path, input_vec):
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        tensor = torch.FloatTensor(input_vec)
        probs = model(tensor)
        predicted = probs.argmax(dim=1)

        for vec, prob, pred in zip(input_vec, probs, predicted):
            print(f"输入：{vec}")
            print(f"各类别概率：{prob.numpy().round(4)}")
            print(f"预测类别：{pred.item()}\n")

if __name__ == "__main__":
    main()
