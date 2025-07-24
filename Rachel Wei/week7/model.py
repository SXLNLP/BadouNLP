# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
import logging

logger = logging.getLogger(__name__)

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        ## encoders
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失
        self.dropout = nn.Dropout(p=config.get("dropout", 0.5))  # default 0.5

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        logger.debug(f"whether to use bert: {self.use_bert}")
        logger.debug(f"model input's shape: {x.shape}")
        logger.debug(f"input looks like this: {x}")

        ## embedding + encoding
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            #sequence_output:batch_size, max_len, hidden_size
            #pooler_output:batch_size, hidden_size
            x = self.encoder(x)
            logger.debug(f"bert encoder shape: {x.shape}")
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len) --> output shape: (batch_size, sen_len, hidden_size)
            x = self.dropout(x) # dropout added after both embedding and encoding
            logger.debug(f"non-bert embedding shape: {x.shape}")
            x = self.encoder(x)  # input shape:(batch_size, sen_len, hidden_size) --> output shape: (batch_size, sen_len, hidden_size)

        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
            logger.debug(f"non-bert encoder shape: {x.shape}")
        
        x = self.dropout(x) # dropout added after both embedding and encoding

        ## Pooling
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
            """ x.shape[1] here means: on the 3rd dimension of the tensor, 
                use a sliding window of size x.shape[1] to do a pooling and give out 1 output
                this size of the sliding window is called kernel_size
            """
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze(-1) 
        logger.debug(f"pooling result's shape: {x.shape}")
        logger.debug(f"pooling result: {x}")
        """input shape:(batch_size, sen_len, hidden_size), 
           output shape (batch_size, hidden_size) sen_len is squeezed to -> 1)
        """


        ### We can also get the sentence vector by doing the following:
        # x = x[:, -1, :]
        
        predict = self.classify(x)   #input shape:(batch_size, hidden_size)
        logger.debug(f"predict result's shape: {predict.shape}")
        logger.debug(f"predict result: {predict}")
        """ predict is defined as nn.Linear(hidden_size, class_num)
        """

        if target is not None:
            logger.debug(f"given sensitivity score: {target}")
            return self.loss(predict, target.view(-1))
        else:
            return predict



class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)
        """
        1. 把 (batch_size, seq_len, hidden) 转成 (batch_size, hidden, seq_len)，
        以便匹配 Conv1d 的输入要求；
        2. 走 self.cnn.forward() 做卷积；
        3. 再转回来，维度变回 (batch_size, seq_len, hidden)。
        """

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        #仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)               #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1) #二层线性
            x = self.bn_after_ff[i](x + l2)        #残差后过bn
        return x


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x

class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]#(13, batch, len, hidden)
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    # Config["class_num"] = 3
    # Config["vocab_size"] = 20
    # Config["max_length"] = 5
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))


    # model = TorchModel(Config)
    # label = torch.LongTensor([1,2])
    # print(model(x, label))