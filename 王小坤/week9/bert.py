import torch
import torch.nn as nn
from transformers import BertModel

class BertForNER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classifier = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        self.use_crf = config.get("use_crf", False)
        if self.use_crf:
            from torchcrf import CRF
            self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1)
                return -self.crf_layer(logits, labels, mask, reduction="mean")
            else:
                return self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(logits)
            else:
                return logits

# 示例config
Config = {
    "bert_path": "bert-base-chinese",  # 或本地bert模型路径
    "class_num": 10,                   # 标签类别数
    "use_crf": False                   # 是否使用CRF
}

if __name__ == "__main__":
    model = BertForNER(Config)
    print(model)
