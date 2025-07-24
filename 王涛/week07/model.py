import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config['vocab_size'] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size,hidden_size,padding_idx=0)
        # 选择模型
        if model_type == "fast_text":
            self.encoder = lambda x:x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size,hidden_size,num_layers=num_layers,batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size,hidden_size,num_layers=num_layers,batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size,hidden_size,num_layers=num_layers,batch_first=True)
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
    def forward(self,x,target=None):
        if self.use_bert:
            # bert返回的结果是 (sequence_output, pooler_output)
            #sequence_output:batch_size, max_len, hidden_size
            #pooler_output:batch_size, hidden_size
            x = self.encoder(x)
        else:
            # input shape:(batch_size, sen_len)
            x = self.embedding(x)
            # input shape:(batch_size, sen_len, input_dim)
            x = self.encoder(x)
        if isinstance(x,tuple):
            x = x[0]
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])

        x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)
        #也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]
        predict = self.classify(x)   #input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size-1)/2)
        self.cnn = nn.Conv1d(hidden_size,hidden_size,kernel_size,bias=False,padding=pad)
    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)
    def forward(self,x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a,b)

class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.gnn_layers = nn.ModuleList(
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
        for i in range(self.num_layers):
            gcnn_x = self.gnn_layers[i](x)
            # 残差
            x = x + gcnn_x
            x = self.bn_after_gcnn[i](x)
            l1 = self.ff_liner_layers1[i](x)
            l1 = torch.relu(l1)
            l2 = self.ff_liner_layers2[i](l1)
            x = self.bn_after_ff(x + l2)
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
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'],return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)
    def forward(self,x):
        x = self.bert(x)[0]
        x , _ = self.rnn(x)
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


