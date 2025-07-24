import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torchcrf import CRF
from config import Config


class TorchModel(nn.Module):
    def __init__(self,config):
        super(TorchModel,self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size,hidden_size,padding_idx=0)
        self.layer = nn.LSTM(hidden_size,hidden_size, batch_first=True,bidirectional=True,num_layers=num_layers)
        self.classify = nn.Linear(hidden_size * 2 ,class_num)
        self.crf_layer = CRF(class_num,batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,x,target =None):
        x = self.embedding(x)
        x,_ = self.layer(x)
        predict = self.classify(x)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return -self.crf_layer(predict,target,mask,reduction="mean")
            else:
                return self.loss(predict.view(-1,predict.shape[-1]),target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict

def choose_optimizer(config,model):
    optimizer = config["optimizer"]
    learning_rate =config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(),lr=learning_rate)
    elif optimizer =="sgd":
        return SGD(model.parameters(),lr=learning_rate)

if __name__ =="__main__":
    model = TorchModel(Config)
