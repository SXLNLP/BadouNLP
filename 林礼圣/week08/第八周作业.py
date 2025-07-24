#week08作业
#修改表示形文本匹配代码，使用三元组损失函数训练。

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.classify = nn.Linear(hidden_size, class_num)
        self.pool = nn.AvgPool1d(max_length)
        self.activation = torch.relu
        self.dropout = nn.Dropout(0.1)
        
        # 不再使用交叉熵损失，改为三元组损失
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, anchor, positive, negative=None):
        """
        前向传播
        anchor: 锚点样本 (batch_size, seq_len)
        positive: 正样本 (batch_size, seq_len)
        negative: 负样本 (batch_size, seq_len)，训练时需要，推理时不需要
        """
        # 获取嵌入表示
        def get_embedding(x):
            x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
            x = self.layer(x)      # (batch_size, seq_len, hidden_size)
            x = self.activation(x)
            x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
            x = self.pool(x).squeeze()  # (batch_size, hidden_size)
            return x
        
        anchor_emb = get_embedding(anchor)
        positive_emb = get_embedding(positive)
        
        # 训练模式需要计算三元组损失
        if negative is not None:
            negative_emb = get_embedding(negative)
            loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
            return anchor_emb, positive_emb, negative_emb, loss
        
        # 推理模式只返回嵌入向量
        return anchor_emb, positive_emb

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

# 辅助函数：生成三元组数据
def generate_triplets(dataset, batch_size=32):
    """
    生成三元组数据 (anchor, positive, negative)
    dataset: 原始数据集，假设每条数据有(text, label)
    """
    # 按标签分组样本
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    triplets = []
    for label, indices in label_to_indices.items():
        # 确保每个类别至少有两个样本
        if len(indices) < 2:
            continue
            
        # 为当前类别生成三元组
        for i in range(len(indices)):
            # 随机选择一个正样本
            j = np.random.choice([k for k in range(len(indices)) if k != i])
            anchor_idx = indices[i]
            positive_idx = indices[j]
            
            # 随机选择一个负样本（不同类别）
            neg_label = np.random.choice([l for l in label_to_indices.keys() if l != label])
            negative_idx = np.random.choice(label_to_indices[neg_label])
            
            triplets.append((anchor_idx, positive_idx, negative_idx))
    
    # 随机打乱三元组
    np.random.shuffle(triplets)
    
    # 分批处理
    for i in range(0, len(triplets), batch_size):
        batch_triplets = triplets[i:i+batch_size]
        anchor_batch = [dataset[idx][0] for idx, _, _ in batch_triplets]
        positive_batch = [dataset[idx][0] for _, idx, _ in batch_triplets]
        negative_batch = [dataset[idx][0] for _, _, idx in batch_triplets]
        
        yield anchor_batch, positive_batch, negative_batch

# 训练函数
def train_model(model, optimizer, train_data, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        
        # 生成三元组批次
        for anchor, positive, negative in generate_triplets(train_data):
            # 转换为张量（这里假设已经预处理为索引序列）
            anchor_tensor = torch.tensor(anchor, dtype=torch.long)
            positive_tensor = torch.tensor(positive, dtype=torch.long)
            negative_tensor = torch.tensor(negative, dtype=torch.long)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播计算损失
            _, _, _, loss = model(anchor_tensor, positive_tensor, negative_tensor)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 相似度计算函数
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

# 推理函数
def predict_match(model, text1, text2):
    """
    预测两个文本是否匹配
    返回: 相似度分数 (0-1之间)
    """
    model.eval()
    with torch.no_grad():
        # 预处理文本为序列（这里简化为占位符）
        seq1 = preprocess_text(text1)
        seq2 = preprocess_text(text2)
        
        # 转换为张量并添加批次维度
        seq1_tensor = torch.tensor([seq1], dtype=torch.long)
        seq2_tensor = torch.tensor([seq2], dtype=torch.long)
        
        # 获取嵌入向量
        emb1, emb2 = model(seq1_tensor, seq2_tensor)
        
        # 计算余弦相似度
        similarity = cosine_similarity(emb1, emb2)
        return similarity.item()

if __name__ == "__main__":
    from config import Config
    
    # 实例化模型
    config = Config()
    model = TorchModel(config)
    optimizer = choose_optimizer(config, model)
    
    # 假设train_data是预处理好的三元组数据集
    # 格式: [(text_seq1, label1), (text_seq2, label2), ...]
    train_data = [...]  # 这里需要填充实际数据
    
    # 训练模型
    train_model(model, optimizer, train_data, epochs=config["epochs"])
    
    # 测试匹配
    text1 = "这款手机电池续航很好"
    text2 = "手机的电池能用一整天"
    similarity = predict_match(model, text1, text2)
    print(f"文本相似度: {similarity:.4f}")
