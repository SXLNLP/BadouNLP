# coding:utf8

import torch
import torch.nn as nn
import os

from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader

"""
把LSTM换成bert来训练
"""


class Config:
    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.corpus_path = 'corpus.txt'
        self.seq_len = 64
        self.batch_size = 16
        self.lr = 1e-5
        self.epochs = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


class SelfDataset(Dataset):
    def __init__(self, tokenizer, corpus_path, seq_len):
        with open(corpus_path, 'r', encoding='gbk') as f:
            text = f.read().replace('\n', '')
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        self.inputs = []
        for i in range(0, len(tokens) - seq_len):
            seq = tokens[i:i + seq_len]
            self.inputs.append(torch.tensor(seq))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx][:-1]  # 输入
        y = self.inputs[idx][1:]  # 标签：预测下一个token
        return x, y


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_config = BertConfig.from_pretrained(config.model_name)
        bert_config.is_decoder = True
        self.bert = BertModel.from_pretrained(config.model_name, config=bert_config)
        self.lm_head = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


def generate_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    return mask  # (1, seq_len, seq_len)


# 文本生成测试代码
def generate_sentence(model, tokenizer, start_text, max_len=50):
    model.eval()
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(start_text))
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(config.device)

    for _ in range(max_len):
        attention_mask = generate_causal_mask(input_ids.size(1)).to(config.device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    result = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return ''.join(result)


def train():
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    dataset = SelfDataset(tokenizer, config.corpus_path, config.seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = LanguageModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for step, (x, y) in enumerate(dataloader):
            x = x.to(config.device)
            y = y.to(config.device)

            attention_mask = generate_causal_mask(x.size(1)).to(config.device)
            logits = model(x, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} finished, Avg Loss: {total_loss / len(dataloader):.4f}")

        print("\n生成效果展示：")
        print("1.", generate_sentence(model, tokenizer, "让他在半年之前，就不能做出"))
        print("2.", generate_sentence(model, tokenizer, "李慕站在山路上，深深的呼吸"))

    torch.save(model.state_dict(), 'causal_bert_lm.pt')
    print("Training complete and model saved.")


if __name__ == "__main__":
    if not os.path.exists('causal_bert_lm.pt'):
        train()
