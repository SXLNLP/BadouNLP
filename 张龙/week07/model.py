import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertModelClass(nn.Module):
   def __init__(self,config):
       super(BertModelClass, self).__init__()
       self.bert = BertModel.from_pretrained('bert-base-cased')
       self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
       self.dropout = nn.Dropout(config.dropout_rate)

   def forward(self,input_ids,attention_mask):
       outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)

       cls_output = outputs.last_hidden_state[:, 0, :]  # 选择 [CLS] token 的表示
       cls_output = self.dropout(cls_output)
       logits = self.classifier(cls_output)
       return logits
# 定义 LSTM 模型
class LSTMModelClass(nn.Module):
    def __init__(self, config, vocab_size):
        super(LSTMModelClass, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)  # 嵌入层
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size * 2, 2)  # 双向 LSTM 输出的隐藏状态需要乘2
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  # 将 input_ids 转化为词嵌入
        lstm_out, _ = self.lstm(embedded)  # 传入 LSTM 网络
        lstm_out = lstm_out[:, -1, :]  # 获取最后一个时间步的输出
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits


# 定义 RNN 模型
class RNNModelClass(nn.Module):
    def __init__(self, config, vocab_size):
        super(RNNModelClass, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)  # 嵌入层
        self.rnn = nn.RNN(config.embedding_dim, config.hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size * 2, 2)  # 双向 RNN 输出的隐藏状态需要乘2
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  # 将 input_ids 转化为词嵌入
        rnn_out, _ = self.rnn(embedded)  # 传入 RNN 网络
        rnn_out = rnn_out[:, -1, :]  # 获取最后一个时间步的输出
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out)
        return logits


# 根据配置选择模型
def get_model(config, vocab_size=None):
    if config.model_type == 'bert':
        return BertModelClass(config)
    elif config.model_type == 'lstm':
        return LSTMModelClass(config, vocab_size)
    elif config.model_type == 'rnn':
        return RNNModelClass(config, vocab_size)
    else:
        raise ValueError("Unsupported model type: choose from 'bert', 'lstm', or 'rnn'.")
