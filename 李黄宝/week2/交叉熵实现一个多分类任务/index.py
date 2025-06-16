import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def build_sample():
    x = np.random.random(5)
    maxNum = x[0]
    maxIndex = 0
    for index,num in enumerate(x):
        if num > maxNum:
            maxNum = num
            maxIndex = index
    return x, maxIndex

def build_dataset(rangeNum):
    xArr, yArr = [], []
    for i in range(rangeNum):
        x, y = build_sample()
        xArr.append(x)
        yArr.append(y)
    return torch.FloatTensor(xArr), torch.LongTensor(yArr)

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear(x)
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(x, y) 
        else:
            return x

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计各类别样本数量
    counts = [0, 0, 0, 0, 0]
    for typeNum in y:
        counts[typeNum] += 1
    print("学习数据的分类概括:", counts)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        # 获取预测类别（最大值的索引）
        predicted = torch.argmax(y_pred, dim=1)
        
        # 计算正确率
        correct = (predicted == y).sum().item()
        wrong = test_sample_num - correct
    
    print("正确预测个数：%d，正确率：%d" % (correct, int(correct / test_sample_num * 100)))
    return correct / test_sample_num

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    output_size = 5 
    learning_rate = 0.001
    model = TorchModel(input_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    
    torch.save(model.state_dict(), "model.bin")
    return

def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    with torch.no_grad():
        logits = model.forward(torch.FloatTensor(input_vec))
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
    
    for vec, prob, pred_class in zip(input_vec, probabilities, predicted_classes):
        print(f"输入：{vec}")
        print(f"类别概率：{[f'{p:.4f}' for p in prob.numpy()]}")
        print(f"预测类别：{pred_class.item()}，最大概率：{torch.max(prob).item():.4f}")
        print("------")

if __name__ == "__main__":
    main()
    test_vec = [
        [0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
        [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
        [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
        [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]
    ]
    predict("model.bin", test_vec)
