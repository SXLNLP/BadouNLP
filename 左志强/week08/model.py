import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np

"""
建立网络模型结构（使用三元组损失）
"""

class TripletModel(nn.Module):
    def __init__(self, config):
        super(TripletModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.margin = config["margin"]  # 三元组损失的边界值
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # 特征提取层
        self.layer = nn.Linear(hidden_size, hidden_size)
        
        # 池化层
        self.pool = nn.AvgPool1d(max_length)
        
        # 激活函数和Dropout
        self.activation = torch.relu
        self.dropout = nn.Dropout(0.1)
        
        # 三元组损失函数
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        
    def get_embedding(self, x):
        """获取输入句子的嵌入表示"""
        # 输入形状: (batch_size, max_length)
        x = self.embedding(x)  # 输出形状: (batch_size, max_length, hidden_size)
        x = self.layer(x)      # 输出形状: (batch_size, max_length, hidden_size)
        x = self.activation(x)  # 应用激活函数
        x = self.dropout(x)     # 应用Dropout
        
        # 池化操作
        x = x.transpose(1, 2)   # 形状变为: (batch_size, hidden_size, max_length)
        x = self.pool(x)        # 输出形状: (batch_size, hidden_size, 1)
        x = x.squeeze(-1)      # 去除最后一个维度，形状: (batch_size, hidden_size)
        return x
    
    def forward(self, anchor, positive, negative):
        """
        前向传播，计算三元组损失
        参数:
            anchor: 锚点样本 [batch_size, max_length]
            positive: 正样本 [batch_size, max_length]
            negative: 负样本 [batch_size, max_length]
        返回:
            三元组损失值
        """
        # 获取嵌入表示
        anchor_embed = self.get_embedding(anchor)
        positive_embed = self.get_embedding(positive)
        negative_embed = self.get_embedding(negative)
        
        # 计算三元组损失
        loss = self.triplet_loss(anchor_embed, positive_embed, negative_embed)
        return loss
    
    def predict(self, x):
        """预测函数，返回输入句子的嵌入表示"""
        with torch.no_grad():
            return self.get_embedding(x)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    # 测试三元组模型
    from config import Config
    
    # 添加三元组损失所需的配置项
    Config["margin"] = 0.5  # 三元组损失的边界值
    
    # 创建模型
    model = TripletModel(Config)
    
    # 创建模拟输入
    batch_size = 2
    max_length = Config["max_length"]
    anchor = torch.randint(0, Config["vocab_size"], (batch_size, max_length))
    positive = torch.randint(0, Config["vocab_size"], (batch_size, max_length))
    negative = torch.randint(0, Config["vocab_size"], (batch_size, max_length))
    
    # 前向传播
    loss = model(anchor, positive, negative)
    print(f"三元组损失值: {loss.item():.4f}")
    
    # 测试预测函数
    embeddings = model.predict(anchor)
    print(f"嵌入向量形状: {embeddings.shape}")
