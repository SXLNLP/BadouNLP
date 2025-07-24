# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        # x, _ = self.lstm(x)  # 如果使用LSTM
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss(margin=0.1, p=2)  # 使用三元组损失函数
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)  # 用于计算余弦相似度

    def cosine_distance(self, tensor1, tensor2):
        return 1 - self.cos(tensor1, tensor2)

    # 输入为三元组 (anchor, positive, negative)
    def forward(self, anchor, positive=None, negative=None, target=None):
        # 同时传入三个句子 (训练模式)
        if positive is not None and negative is not None:
            anchor_vec = self.sentence_encoder(anchor)
            positive_vec = self.sentence_encoder(positive)
            negative_vec = self.sentence_encoder(negative)
            
            # 计算三元组损失
            loss = self.loss(anchor_vec, positive_vec, negative_vec)
            
            # 如果需要计算准确率或其他指标
            if target is not None:
                pos_dist = self.cosine_distance(anchor_vec, positive_vec)
                neg_dist = self.cosine_distance(anchor_vec, negative_vec)
                correct = (pos_dist < neg_dist).float().mean()  # 正样本距离应小于负样本距离
                return loss, correct
            return loss
        
        # 单独传入一个句子时 (推断模式)
        elif positive is None and negative is None:
            return self.sentence_encoder(anchor)
        
        # 传入两个句子时 (评估模式)
        elif negative is None:
            vec1 = self.sentence_encoder(anchor)
            vec2 = self.sentence_encoder(positive)
            return self.cosine_distance(vec1, vec2)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    
    # 测试三元组输入
    anchor = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    positive = torch.LongTensor([[1,2,3,4], [3,2,0,0]])
    negative = torch.LongTensor([[4,3,2,1], [1,3,4,0]])
    
    loss, acc = model(anchor, positive, negative, target=None)
    print(f"Triplet Loss: {loss.item()}, Accuracy: {acc.item()}")
    
    # 测试单个句子输入 (编码)
    vec = model(anchor)
    print(f"Encoded vector shape: {vec.shape}")
    
    # 测试两个句子输入 (相似度计算)
    sim = model(anchor, positive)
    print(f"Similarity scores: {sim}")
