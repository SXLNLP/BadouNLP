# models.py
import torch.nn as nn
from transformers import BertModel
from config import global_config as config


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(config.pretrained_model_name)
        # 添加Dropout防止过拟合
        self.dropout = nn.Dropout(0.1)
        # 添加分类层
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 通过BERT模型
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        # 获取[CLS]标记的隐藏状态作为句子表示
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits