
import numpy as np
import torch
import torch.nn as nn


class CrossEntropyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CrossEntropyModel, self).__init__()

        # self.linear01 = nn.Linear(input_size, hidden_size)
        # self.linear02 = nn.Linear(hidden_size, output_size*2)
        # self.linear03 = nn.Linear(output_size*2, output_size)
        # self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(input_size, output_size)
        # self.active01 = nn.Sigmoid()
        # self.active02 = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target=None):
        # y_pred = self.linear01(input)
        # y_pred = self.active01(y_pred)
        # y_pred = self.linear02(y_pred)
        # y_pred = self.active02(y_pred)
        # y_pred = self.linear03(y_pred)
        # y_pred = self.dropout(y_pred)
        y_pred = self.linear(input)
        if target is not None:
            return self.loss(y_pred, target)
        else:
            return torch.softmax(y_pred, dim=-1)


def build_sample(type_size):
    # y_t = np.zeros(type_size)
    # x = np.random.randn(type_size)
    # type_index = np.argmax(x, axis=-1)
    # y_t[type_index] = 1
    # x = np.random.randn(type_size)
    x = np.random.random(type_size)
    y_t = np.argmax(x, axis=-1)
    return x, y_t

def build_dataset(category_size, total_amount):
    dataset_x = []
    dataset_y = []
    for i in range(total_amount):
        x, y = build_sample(category_size)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.FloatTensor(dataset_x), torch.LongTensor(dataset_y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    category_size = 5
    test_sample_num = 100
    x, y = build_dataset(category_size, test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():  # 用于在其上下文块中禁用自动求导（Autograd）的功能，从而避免梯度计算和反向传播(反向传播会影响权重)
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
            # if torch.argmax(y_p) == torch.argmax(y_t):
            #     correct += 1
            # else:
            #     wrong += 1
    correct_rate = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, correct_rate))
    return correct_rate

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = CrossEntropyModel(5, 20, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        data_type = torch.argmax(res) + 1
        print("输入：%s, 预测类别：%d" % (vec, round(float(data_type))))  # 打印结果

def main():
    epochs = 20  # 训练轮数
    batch_size = 20  # 批量训练的样本数量
    sample_amount = 20000
    type_size = 5  # 类型数量
    model = CrossEntropyModel(input_size=type_size, hidden_size=type_size*4, output_size=type_size)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_x, train_y = build_dataset(type_size, sample_amount)
    for epoch in range(epochs):
        model.train()
        loss_watch = []
        for batch_num in range(sample_amount // batch_size):
            x = train_x[batch_num * batch_size: batch_num * batch_size + batch_size]
            y = train_y[batch_num * batch_size: batch_num * batch_size + batch_size]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            loss_watch.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(loss_watch)))
        evaluate(model)
    torch.save(model.state_dict(), 'week2_weight_path.pth')


if __name__ == '__main__':
    main()
    test_vec = [[0.47889086, 0.75229675, 0.31082123, 0.03504317, 0.84453329],  # 5
                [0.94963533, 0.5524256, 0.35758807, 0.95520434, 0.34453329],  # 4
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.34453329],  # 1
                [0.1349776, 0.59416669, 0.92579291, 0.41567412, 0.34453329],  # 3
                [0.78797868, 0.9482528, 0.43625847, 0.34675372, 0.34453329],  # 2
                [0.78797868, 0.67482528, 0.83625847, 0.34675372, 0.34453329]]  # 3
    predict("week2_weight_path.pth", test_vec)