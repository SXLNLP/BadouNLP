import random
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置随机种子确保可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 文本预处理函数
def preprocess_text(text):
    """清洗和标准化文本"""
    # 转换为小写
    text = text.lower()
    # 移除特殊字符和数字
    text = re.sub(r"[^a-z\s]", "", text)
    # 移除多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 加载示例数据集 (20 Newsgroups)
print("Loading and preprocessing dataset...")
newsgroups = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes'),
    categories=['sci.space', 'rec.sport.baseball', 'comp.graphics']
)
texts = [preprocess_text(text) for text in newsgroups.data]
labels = newsgroups.target

# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)
print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")

# 构建词汇表
print("Building vocabulary...")
word_counter = Counter()
for text in train_texts:
    word_counter.update(text.split())

# 创建词汇表（只保留最常见的单词）
VOCAB_SIZE = 10000
vocab = {
    word: idx + 2 for idx, (word, count) in enumerate(word_counter.most_common(VOCAB_SIZE - 2))
}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
reverse_vocab = {idx: word for word, idx in vocab.items()}

print(f"Vocabulary size: {len(vocab)}")


# 文本编码函数
def text_to_sequence(text, max_length=128):
    """将文本转换为整数序列"""
    tokens = text.split()
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    elif len(sequence) < max_length:
        sequence += [vocab["<PAD>"]] * (max_length - len(sequence))
    return sequence


# 三元组数据集类
class TripletDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.label_to_indices = self._build_label_index()

    def _build_label_index(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __getitem__(self, index):
        anchor_text = self.texts[index]
        anchor_label = self.labels[index]

        # 正样本：同标签随机选 (确保不选到自己)
        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_text = self.texts[positive_idx]

        # 负样本：不同标签随机选
        negative_label = random.choice(list(set(self.labels) - {anchor_label}))
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_text = self.texts[negative_idx]

        # 转换为序列
        anchor_seq = text_to_sequence(anchor_text)
        positive_seq = text_to_sequence(positive_text)
        negative_seq = text_to_sequence(negative_text)

        return (
            torch.tensor(anchor_seq, dtype=torch.long),
            torch.tensor(positive_seq, dtype=torch.long),
            torch.tensor(negative_seq, dtype=torch.long)
        )

    def __len__(self):
        return len(self.texts)


# 文本编码器模型 
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 输入x: [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        _, (hidden, _) = self.lstm(embedded)  # hidden: [num_layers, batch_size, hidden_dim]

        # 取最后一层的隐藏状态
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]

        # 全连接层
        output = self.fc(last_hidden)
        output = self.dropout(output)
        return output


# 三元组损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, distance='euclidean'):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative):
        if self.distance == 'euclidean':
            pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
            neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        elif self.distance == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError("Unsupported distance metric")

        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


# 创建数据集和数据加载器
train_dataset = TripletDataset(train_texts, train_labels)
test_dataset = TripletDataset(test_texts, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4 if torch.cuda.is_available() else 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4 if torch.cuda.is_available() else 0
)

# 初始化模型、损失函数和优化器
model = TextEncoder(len(vocab)).to(device)
criterion = TripletLoss(margin=0.5, distance='euclidean')
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for anchor, positive, negative in progress_bar:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()

        # 获取文本向量
        anchor_vec = model(anchor)
        positive_vec = model(positive)
        negative_vec = model(negative)

        # 计算损失
        loss = criterion(anchor_vec, positive_vec, negative_vec)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


# 评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_vec = model(anchor)
            positive_vec = model(positive)
            negative_vec = model(negative)

            loss = criterion(anchor_vec, positive_vec, negative_vec)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# 训练循环
NUM_EPOCHS = 15
train_losses = []
test_losses = []

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # 更新学习率
    scheduler.step(test_loss)

    # 保存模型检查点
    if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'vocab': vocab
        }, f"model_checkpoint_epoch_{epoch}.pth")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()

# 保存最终模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab': vocab,
    'config': {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 1
    }
}, "text_encoder_model.pth")
print("Model saved to text_encoder_model.pth")


# 测试相似度计算函数
def calculate_similarity(model, text1, text2):
    model.eval()
    with torch.no_grad():
        seq1 = text_to_sequence(preprocess_text(text1))
        seq2 = text_to_sequence(preprocess_text(text2))

        vec1 = model(torch.tensor([seq1], dtype=torch.long).to(device))
        vec2 = model(torch.tensor([seq2], dtype=torch.long).to(device))

        similarity = F.cosine_similarity(vec1, vec2).item()
    return similarity


# 示例相似度计算
sample_text1 = "The space shuttle launched successfully yesterday."
sample_text2 = "NASA announced a new mission to Mars next year."
sample_text3 = "The baseball game was postponed due to rain."

sim_related = calculate_similarity(model, sample_text1, sample_text2)
sim_unrelated = calculate_similarity(model, sample_text1, sample_text3)

print("\nSimilarity between related texts:", sim_related)
print("Similarity between unrelated texts:", sim_unrelated)


# 测试模型在同类文本上的表现
def test_model_on_same_class(model, dataset):
    model.eval()
    similarities = []

    with torch.no_grad():
        for i in range(100):  # 测试100个样本
            # 随机选择一个类别
            label = random.choice(list(dataset.label_to_indices.keys()))
            indices = dataset.label_to_indices[label]

            # 随机选择两个同类别文本
            idx1, idx2 = random.sample(indices, 2)
            text1 = dataset.texts[idx1]
            text2 = dataset.texts[idx2]

            seq1 = text_to_sequence(text1)
            seq2 = text_to_sequence(text2)

            vec1 = model(torch.tensor([seq1], dtype=torch.long).to(device))
            vec2 = model(torch.tensor([seq2], dtype=torch.long).to(device))

            similarity = F.cosine_similarity(vec1, vec2).item()
            similarities.append(similarity)

    return np.mean(similarities)


# 测试模型在不同类文本上的表现
def test_model_on_different_class(model, dataset):
    model.eval()
    similarities = []

    with torch.no_grad():
        for i in range(100):  # 测试100个样本
            # 随机选择两个不同类别
            label1, label2 = random.sample(list(dataset.label_to_indices.keys()), 2)

            # 从每个类别随机选择一个文本
            text1 = dataset.texts[random.choice(dataset.label_to_indices[label1])]
            text2 = dataset.texts[random.choice(dataset.label_to_indices[label2])]

            seq1 = text_to_sequence(text1)
            seq2 = text_to_sequence(text2)

            vec1 = model(torch.tensor([seq1], dtype=torch.long).to(device))
            vec2 = model(torch.tensor([seq2], dtype=torch.long).to(device))

            similarity = F.cosine_similarity(vec1, vec2).item()
            similarities.append(similarity)

    return np.mean(similarities)


# 运行测试
same_class_sim = test_model_on_same_class(model, train_dataset)
diff_class_sim = test_model_on_different_class(model, train_dataset)

print(f"\nAverage similarity within same class: {same_class_sim:.4f}")
print(f"Average similarity between different classes: {diff_class_sim:.4f}")
