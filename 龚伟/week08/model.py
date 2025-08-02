import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        # 自定义如何将文本向量化，例如先embedding后线性层
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        # 再pooling
        # 也可以先lstm再pooling
        # 输入文本转成向量
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

# 训练文本匹配，单独建一个模型，把上面的模型引入进来
# 孪生网络
class SiameseNetwork(nn.Module):
    # 模型主体：引入了一个sentenceEncoder 在上面定义了
    #
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        # cos 做一个映射，使他成为合格的loss
        self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    # 使用了cos embedding loss
    #前向 传入向量
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        # 计算a和p的余弦距离
        # 计算a和n的余弦距离
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
        # 如果没设置margin，则diff为ap - an + 0.1
            diff = ap - an + 0.1
        else:
        # 如果设置了margin，则diff为ap - an + margin.squeeze()
            diff = ap - an + margin.squeeze()
        # 返回diff中大于0的部分的平均值
        return torch.mean(diff[diff.gt(0)]) #greater than

    #sentence : (batch_size, max_length)
    # 如果用triplet loss训练的话，forward需要改
    # 传入三句话，并将三句话都向量化
    # 三个向量传入cos_triplet_loss去计算
    # def forward(self, sentence1, sentence2=None, target=None):
    #     #同时传入两个句子
    #     # 对传入的sentence1和sentence2分别向量化
    #     if sentence2 is not None:
    #         vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)
    #         # 可以在这里写要不要把vector1和vector2拼接
    #         #如果有标签，则计算loss
    #         # 将标签一并传入，标签是+1或-1 cos embedding loss 要求
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         #如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     #单独传入一个句子时，认为正在使用向量化能力
    #     # 只传入一个句子，就执行正常的向量化
    #     # 如果同时传入了两个句子，有target就算loss
    #     # 没target就直接算cos距离
    #     else:
    #         return self.sentence_encoder(sentence1)
    def forward(self, anchor, positive, negative):
        # 对锚点、正样本和负样本进行向量化
        anchor_vector = self.sentence_encoder(anchor)  # vec: (batch_size, hidden_size)
        positive_vector = self.sentence_encoder(positive)
        negative_vector = self.sentence_encoder(negative)

        # 计算三元组损失
        loss = self.cosine_triplet_loss(anchor_vector, positive_vector, negative_vector)

        return loss


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
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())
