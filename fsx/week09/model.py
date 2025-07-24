import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from torch.optim import Adam, SGD


class NerBertModel(nn.Module):
    def __init__(self, config):
        super(NerBertModel, self).__init__()
        self.use_bert = config["use_bert_switch"]
        bert_path = config.get("bert_path", "")
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        class_num = config["class_num"]
        self.pad_token_id = config.get("pad_token_id", 0)  # 统一padding id（通常为0）
        self.ignore_label = config.get("ignore_label", -1)  # 忽略的标签值

        if self.use_bert:
            self.encoder = BertModel.from_pretrained(bert_path, return_dict=False)
            encoder_dim = self.encoder.config.hidden_size
            self.classify = nn.Linear(encoder_dim, class_num)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_label)  # 忽略无效标签
        else:
            # LSTM模型 + CRF
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=self.pad_token_id)
            self.lstm = nn.LSTM(
                hidden_size, hidden_size,
                batch_first=True,
                bidirectional=True,
                num_layers=num_layers
            )
            encoder_dim = hidden_size * 2  # 双向LSTM输出维度
            self.classify = nn.Linear(encoder_dim, class_num)
            self.crf = CRF(class_num, batch_first=True)

        # 正则化
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, target=None):
        if self.use_bert:
            outputs = self.encoder(x, return_dict=False)
            sequence_output = self.dropout(outputs[0])
            logits = self.classify(sequence_output)

            if target is not None:
                batch_size, seq_len, num_classes = logits.size()
                logits_reshaped = logits.view(-1, num_classes)
                target_reshaped = target.view(-1)
                return self.loss_fn(logits_reshaped, target_reshaped)
            else:
                # 推理时仅返回有效位置的预测
                return torch.argmax(logits, dim=-1)
        else:
            # LSTM+CRF模式
            embedded = self.embedding(x)  # [batch, seq_len, hidden_size]
            lstm_output, _ = self.lstm(embedded)  # [batch, seq_len, 2*hidden_size]
            sequence_output = self.dropout(lstm_output)
            emissions = self.classify(sequence_output)  # [batch, seq_len, class_num]

            if target is not None:
                mask = target != self.ignore_label  # 基于target的有效标签生成mask
                return -self.crf(emissions, target, mask=mask, reduction="mean")
            else:
                mask = x != self.pad_token_id  # 推理时基于x的padding生成mask
                return self.crf.decode(emissions, mask=mask)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from config import Config

    model = NerBertModel(Config)

    # 测试数据
    x = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
    print(model(x))
