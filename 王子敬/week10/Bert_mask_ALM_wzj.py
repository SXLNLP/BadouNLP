
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer


class LanguageModel(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.classify = nn.Linear(self.bert.config.hidden_size, len(self.tokenizer.vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # 基础padding掩码
        attention_mask = (x != self.tokenizer.pad_token_id).float()

        # 添加因果掩码（Causal Mask）使其只能看到左侧
        batch_size, seq_len = x.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)  # 下三角矩阵
        attention_mask = attention_mask.unsqueeze(1) * causal_mask  # 合并padding掩码和因果掩码

        # BERT前向传播（关键修改：传入自定义attention_mask）
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,  # 使用单向掩码
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        y_pred = self.classify(sequence_output)

        if y is not None:
            loss = self.loss(
                y_pred.view(-1, y_pred.shape[-1]),
                y.view(-1)
            )
            return loss
        else:
            return torch.softmax(y_pred, dim=-1)


def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    window = corpus[start:start + window_size]
    target = corpus[start + 1:start + window_size + 1]

    encoded_input = tokenizer(
        window,
        padding='max_length',
        max_length=window_size,
        truncation=True,
        return_tensors='pt'
    )
    encoded_target = tokenizer(
        target,
        padding='max_length',
        max_length=window_size,
        truncation=True,
        return_tensors='pt'
    )
    return encoded_input['input_ids'].squeeze(0), encoded_target['input_ids'].squeeze(0)


def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x, dataset_y = [], []
    for _ in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.stack(dataset_y)


def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        while len(openings) <= 30:
            encoded = tokenizer(
                openings[-window_size:],
                padding='max_length',
                max_length=window_size,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            logits = model(input_ids)[0][-1]
            next_token_id = sampling_strategy(logits)
            next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
            if next_token in ["[SEP]", "\n"]:
                break
            openings += next_token
    return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        return int(torch.argmax(prob_distribution))
    else:
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(len(prob_distribution), p=prob_distribution)


def train(corpus_path, save_weight=False):
    epoch_num = 20
    batch_size = 128
    train_sample = 50000
    window_size = 10
    bert_path = r"D:\BaiduYunDownload\八斗精品班\bert-base-chinese"

    model = LanguageModel(bert_path)
    tokenizer = model.tokenizer
    corpus = load_corpus(corpus_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    print("模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            x, y = x.to(device), y.to(device)

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()

            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train("corpus.txt", False)
  
