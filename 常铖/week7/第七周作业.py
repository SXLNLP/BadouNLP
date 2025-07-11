import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 定义一个简单的嵌入层
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 32)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 三元组损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# 创建一个虚拟的数据集
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, embedding_dim=100):
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.samples = torch.randn(num_samples, embedding_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anchor = self.samples[idx]
        positive_idx = torch.randint(0, self.num_samples, (1,))
        while positive_idx == idx:
            positive_idx = torch.randint(0, self.num_samples, (1,))
        positive = self.samples[positive_idx]
        negative_idx = torch.randint(0, self.num_samples, (1,))
        while negative_idx == idx or negative_idx == positive_idx:
            negative_idx = torch.randint(0, self.num_samples, (1,))
        negative = self.samples[negative_idx]
        return anchor, positive, negative

# 初始化数据加载器和模型
dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = EmbeddingNet()
criterion = TripletLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (anchors, positives, negatives) in enumerate(dataloader):
        optimizer.zero_grad()
        anchors_out = model(anchors)
        positives_out = model(positives)
        negatives_out = model(negatives)
        loss = criterion(anchors_out, positives_out, negatives_out)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
