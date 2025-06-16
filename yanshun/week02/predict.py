import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # 线性层
        self.linear2 = nn.Linear(hidden_size, 5)
        self.sig = nn.Sigmoid()  # nn.Sigmoid() sigmoid归一化函数
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 1)
        x = self.sig(x)  # (batch_size, 1) -> (batch_size, 1)
        x = torch.softmax(x, dim=1)
        x = self.linear2(x)
        y_pred = x
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    参数说明：
    cm : 计算好的混淆矩阵
    classes : 类别名称列表
    normalize : 是否归一化显示
    title : 图表标题
    cmap : 颜色映射
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("归一化混淆矩阵")
    else:
        print('未归一化混淆矩阵')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

def predict_batch(model_path, data_path):
    input_size = 5
    hidden_size = 5
    model = TorchModel(input_size, hidden_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()
    loaded_data = torch.load('dataset.pt', weights_only=False)
    x, y = loaded_data['X'][9000:], loaded_data['Y'][9000:]
    correct, wrong = 0, 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_t[torch.argmax(y_p)] == 1:
                correct += 1  # 分类正确
            else:
                wrong += 1  # 分类错误
            all_preds = torch.argmax(y_pred, dim=1).cpu().numpy()
            all_targets = torch.argmax(y, dim=1).cpu().numpy()
    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong)}")

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)

    # 可视化
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    plot_confusion_matrix(cm, classes=class_names, normalize=True)
    return correct / (correct + wrong)

if __name__ == "__main__":
    model_path = "model99.bin"
    data_path = "dataset.pt"
    predict_batch(model_path, data_path)