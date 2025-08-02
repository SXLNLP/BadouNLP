import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络bert模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1 #为词汇表大小增加了一个padding索引
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.bert = BertModel.from_pretrained(config["bert_path"],return_dict = False)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0) #指定索引0用于填充位置的零向量化
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers) #双向lstmbidirectional=True 使LSTM输出包含前向和后向隐藏状态的拼接结果
        self.classify = nn.Linear(self.bert.config.hidden_size, self.class_num)
        self.crf_layer = CRF(self.class_num, batch_first=True)
        self.use_crf = config["use_crf"] #use_crf 开关允许灵活选择是否使用CRF解码
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # sequence_output,pooler_output = self.bert(x)
        if not torch.is_tensor(x):
            x = torch.tensor(x,dtype=torch.long)
        elif x.dtype != torch.long:
            x = x.long()
        outputs = self.bert(input_ids=x)  # 修正后:ml-citation{ref="1" data="citationList"}
        sequence_output = outputs[0]
        pooler_output = outputs[1]
        predict = self.classify(pooler_output)

        if target is not None:  #用于指定真实标签，形状通常为 (batch_size, sen_len)，当进行训练且需要计算损失时使用
            if self.use_crf:
                mask = target.gt(-1) #通过 target.gt(-1) 生成一个掩码 mask，用于指示有效标签的位置（通常 -1 表示填充或无效标签）。
                return - self.crf_layer(predict, target, mask, reduction="mean") #如果用crf加一个crf层 使用CRF层计算损失，并返回负损失值（因为CRF层通常返回负对数似然损失）
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1)) #不用crf就用交叉熵算loss 将 predict 和 target 重塑为一维张量，并使用交叉熵损失函数计算损
        else: #无 target 时
            if self.use_crf:
                return self.crf_layer.decode(predict)  #解码时如果没有用crf，直接用正常输出的发射矩阵贪婪的方向去解码，因为没有发射矩阵就没有篱笆墙的问题 使用CRF层的解码方法获取最佳标签序列
            else: #如果不使用CRF层：
                return predict #直接返回分类层的输出 predict，通常需要在后续步骤中使用贪婪策略或维特比算法进行解码。


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    import torch
    model = TorchModel(Config)
    test_input = torch.tensor([[1,2,3]])  # 模拟batch_size=1, seq_len=10的输入
    output = model(test_input)
