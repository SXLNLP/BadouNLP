import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.model_selection import train_test_split
import random


# 1. 配置参数
class Config:
    vocab_size = 10000  # 词汇表大小
    embedding_dim = 256
    hidden_dim = 128
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    margin = 0.2  # 三元组损失边界
    max_len = 30  # 最大文本长度


# 2. 三元组数据集类
class TripletTextDataset(Dataset):
    def __init__(self, texts, labels, num_triplets=5000):
        self.texts = texts
        self.labels = labels
        self.label_to_indices = {}

        # 构建标签到索引的映射
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # 生成三元组 (anchor, positive, negative)
        self.triplets = []
        for _ in range(num_triplets):
            # 随机选择一个类别作为锚点
            anchor_label = random.choice(list(self.label_to_indices.keys()))
            # 从同一类中随机选正样本
            pos_idx = random.choice(self.label_to_indices[anchor_label])

            # 随机选择不同类别的负样本
            negative_labels = [l for l in self.label_to_indices.keys() if l != anchor_label]
            negative_label = random.choice(negative_labels)
            neg_idx = random.choice(self.label_to_indices[negative_label])

            # 添加三元组 (anchor, positive, negative)
            self.triplets.append((pos_idx, pos_idx, neg_idx))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        return (
            self.texts[anchor_idx],
            self.texts[pos_idx],
            self.texts[neg_idx]
        )


# 3. 文本编码模型 (LSTM)
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # 合并双向LSTM的最终隐藏状态
        return torch.cat((hidden[-2], hidden[-1]), dim=1)


# 4. 三元组损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)


# 5. 数据预处理和加载
def collate_fn(batch):
    anchor, pos, neg = zip(*batch)

    # 文本转为张量并填充
    def process_texts(texts):
        return pad_sequence(
            [torch.tensor(t) for t in texts],
            batch_first=True,
            padding_value=0
        )

    return (
        process_texts(anchor),
        process_texts(pos),
        process_texts(neg)
    )


# 6. 训练函数
def train_model():
    config = Config()

    # 模拟数据生成 (实际使用时替换为真实数据)
    texts = [np.random.randint(1, config.vocab_size, random.randint(5, config.max_len))
             for _ in range(1000)]
    labels = [random.randint(0, 9) for _ in range(1000)]

    # 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 创建数据集和数据加载器
    train_dataset = TripletTextDataset(train_texts, train_labels, num_triplets=5000)
    test_dataset = TripletTextDataset(test_texts, test_labels, num_triplets=1000)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        collate_fn=collate_fn
    )

    # 初始化模型和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextEncoder(config).to(device)
    criterion = TripletLoss(margin=config.margin)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            # 获取文本嵌入
            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)

            # 计算损失
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            total_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{config.num_epochs}], Loss: {avg_loss:.4f}')

    print("训练完成！")
    return model


# 7. 执行训练
if __name__ == "__main__":
    model = train_model()
