# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module): #只是把文本转成向量
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"] #控制输出向量的维度
        vocab_size = config["vocab_size"] + 1 #词汇表大小（+1用于padding索引0）
        max_length = config["max_length"] #限制输入序列的最大长度
        self.word2idx = config.get("word2idx", {}) #新增词表映射
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0) #文本向量话 #padding_idx=0忽略填充符
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True) #过lstm取最后一维作向量
        self.layer = nn.Linear(hidden_size, hidden_size) #过线性层 对嵌入向量进行线性变换（注释掉的LSTM层可替换为序列建模）
        self.dropout = nn.Dropout(0.5) #防止过拟合的正则化层

    #输入为问题字符编码 这是一个文本向量化的前向传播实现，支持多种神经网络架构的灵活切换。
    def forward(self, x):
        if isinstance(x, str):
            x = self._preprocess_text(x)
        elif not isinstance(x, torch.Tensor):
            x = torch.LongTensor(x)
        x = self.embedding(x) #通过self.embedding将输入文本的单词索引转换为稠密向量表示，形成序列的向量化表达
        #使用lstm
        # x, _ = self.lstm(x) LSTM分支：注释中保留的self.lstm可处理序列依赖关系
        #使用线性层
        x = self.layer(x) #当前启用的self.layer进行线性变换
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()  #pooling层转成向量
        return x



    def _preprocess_text(self, text):
         tokens = text.split()
         indices = [self.word2idx.get(t, 0) for t in tokens]
         return torch.LongTensor([indices])

class SiameseNetwork(nn.Module): #训练文本匹配的模型（孪生网络） 共享权重的句子编码器，将输入文本转换为向量表示
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, vec1,vec2):
        if vec1.dim()==1:
            vec1 = vec1.unsqueeze(0)
        if vec2.dim()==1:
            vec2 = vec2.unsqueeze(0)
        return 1-torch.cosine_similarity(vec1,vec2,dim=1) #最终返回1 - cosine转换为距离度量（值域[0,2]）

    def cosine_triplet_loss(self, a, p, n, margin=0.1): #输入包含锚样本a、正样本p和负样本n
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)])

    #sentence : (batch_size, max_length)
    def forward(self, anchor, positive=None, negative=None,target=None):
        #同时传入两个句子 通过共享权重的sentence_encoder将两个句子分别编码为特征向量vector1和vector2
        if positive is not None and negative is not None:
            a_vec = self.sentence_encoder(anchor)
            p_vec = self.sentence_encoder(positive)
            n_vec = self.sentence_encoder(negative)

            if a_vec.dim()==1:
                a_vec = a_vec.unsqueeze(0)
            if p_vec.dim()==1:
                p_vec = p_vec.unsqueeze(0)
            if n_vec.dim() == 1:
                n_vec = n_vec.unsqueeze(0)
            if target is not None:
                return self.cosine_triplet_loss(a_vec, p_vec, n_vec,target) #传进真正对应的标签1或-1 cosine loss的要求
            return self.cosine_distance(a_vec, p_vec),self.cosine_distance(a_vec, n_vec)
        elif positive is not None: #双句模式
            vec1 = self.sentence_encoder(anchor)
            vec2 = self.sentence_encoder(positive)
            if target is not None:
                return self.cosine_distance(vec1,vec2)
            #如果无标签，计算余弦距离 无标签时直接返回两个向量的余弦距离作为相似度得分
        #单独传入一个句子时，认为正在使用向量化能力
        else: #单句模式
            return self.sentence_encoder(anchor) #直接返回sentence_encoder对单个句子的编码结果

#这段代码实现了一个优化器选择器函数，主要用于根据配置动态创建不同的优化器对象
def choose_optimizer(config, model): #config字典必须包含两个关键字段 "optimizer"：指定优化器类型（如"adam"/"sgd"） "learning_rate"：设置学习率数值 model用于获取待优化参数列表
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd": #当配置为"sgd"时，实例化随机梯度下降优化器
        return SGD(model.parameters(), lr=learning_rate) #两种优化器均使用统一的学习率配置


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10 #动态修改Config字典的vocab_size(词汇表大小)和max_length(最大序列长度)
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    anchor = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    positive = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    negative = torch.LongTensor([[4,3,2,1], [0,0,2,2]])
    loss = model.loss(
        model.sentence_encoder(anchor),
        model.sentence_encoder(positive),
        model.sentence_encoder(negative)
    )
    print(loss)
    # print(model.state_dict())
