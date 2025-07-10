# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from config_homework import Config

class BaseModel(nn.Module):
    def __init__(self, mode='dnn'):
        super(BaseModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config["pretrain_model_path"])
        self.mode = mode.lower()
        hidden_size = Config["hidden_size"]

        if self.mode == 'dnn':
            self.classifier = nn.Sequential(
                nn.Linear(768, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            )
        elif self.mode == 'cnn':
            self.conv = nn.Conv1d(768, hidden_size, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(hidden_size, 2)
        elif self.mode == 'rnn':
            self.rnn = nn.GRU(input_size=768, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 2)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        last_hidden = outputs[0]

        if self.mode == 'dnn':
            cls_vector = last_hidden[:, 0, :]
            return self.classifier(cls_vector)
        elif self.mode == 'cnn':
            x = last_hidden.permute(0, 2, 1)
            x = self.pool(torch.relu(self.conv(x))).squeeze(-1)
            return self.fc(x)
        elif self.mode == 'rnn':
            _, hn = self.rnn(last_hidden)
            return self.fc(hn.squeeze(0))
