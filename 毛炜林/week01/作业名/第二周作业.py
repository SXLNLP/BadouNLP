import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 1、创建数据集
def build_data():
    n = np.random.random(5)
    max_index = np.argmax(n)
    y = max_index
    return n, y


def build_dataset(total_sample):
    X = []
    Y = []
    for i in range(total_sample):
        x, y = build_data()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(X), torch.LongTensor(Y)  # 使用LongTensor存储整数标签

def bulid_testset(total_sample):
    X = []
    for i in range(total_sample):
        x, _ = build_data()
        X.append(x)

    return X


# 2、创建模型 -
class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.fc1(x)
        x = self.relu(x)
        y_pred = self.fc2(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=1)


# 3、训练
def train():
    epochs = 100
    batch_size = 25
    total_sample = 10000
    input_size = 5
    model = Model(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_x, train_y = build_dataset(total_sample)
    log = []

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for i in range(total_sample // batch_size):
            x = train_x[i * batch_size:(i + 1) * batch_size]
            y = train_y[i * batch_size:(i + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"======\n第{epoch + 1}轮平均loss:{avg_loss:.6f}")
        log.append(avg_loss)

    torch.save(model.state_dict(), "model1.bin")

    # 绘制损失曲线
    plt.plot(range(epochs), log)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    return


# 4、预测 
def predict(model_path, input_data):
    input_size = 5
    model = Model(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data)
        pred = model(input_tensor)
        _, predicted_classes = torch.max(pred, 1)

        correct = 0
        worry =0

        for i, (vec, prob, pred) in enumerate(zip(input_data, pred, predicted_classes)):
            print(f"第{i + 1}样本，输入向量{vec} \n"
                  f"预测概率分布{prob.numpy().tolist()}, 预测类别{pred.item()}, 实际最大值位置{np.argmax(vec)}")
            if pred.item() == np.argmax(vec):
                #print("预测正确")
                correct +=1
            else:
                #print("预测错误")
                worry +=1
    print(f"测试集一共%d样本，预测正确%d,预测错误%d" % (correct+worry,correct, worry))


if __name__ == '__main__':
    train()

    test_data = bulid_testset(300)
    predict("model1.bin", test_data)


    '''
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model1.bin", test_vec)
    '''


