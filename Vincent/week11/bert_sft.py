import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, bert_path, freeze_bert=False):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(
            bert_path,
            return_dict=False
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略非答案部分

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden, _ = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classify(hidden)

        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss
        return logits


def load_corpus(path):
    corpus = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                question = item.get('title', '').strip()
                answer = item.get('content', '').strip()
                if question and answer:
                    corpus.append({'question': question, 'answer': answer})
            except json.JSONDecodeError:
                print(f"跳过解析错误行: {line[:50]}...")
    return corpus


def build_dataset(batch_size, tokenizer, max_length, corpus):
    input_ids_list, labels_list, attention_mask_list = [], [], []
    for _ in range(batch_size):
        qa = random.choice(corpus)
        question = qa['question']
        answer = qa['answer']

        q_tokens = tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=max_length // 2)
        a_tokens = tokenizer.encode(answer, add_special_tokens=False, truncation=True, max_length=max_length // 2)

        input_ids = [tokenizer.cls_token_id] + q_tokens + [tokenizer.sep_token_id] + a_tokens + [tokenizer.sep_token_id]
        labels = [-100] * (len(q_tokens) + 2) + a_tokens + [tokenizer.sep_token_id]

        # padding
        pad_len = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask = [1] * (max_length - pad_len) + [0] * pad_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return torch.tensor(input_ids_list), torch.tensor(labels_list), torch.tensor(attention_mask_list)


def build_model(vocab_size, bert_path, freeze_bert=True):
    hidden_size = 768  # bert-base-chinese的hidden size
    return LanguageModel(hidden_size, vocab_size, bert_path, freeze_bert=freeze_bert)


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def generate_sentence(prompt, model, tokenizer, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_id = sampling_strategy(outputs[0, -1])
            if next_token_id == tokenizer.sep_token_id:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(input_ids.device)], dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 64
    train_sample = 50000
    max_length = 128  # 输入序列最大长度
    lr = 0.001
    bert_path = r"D:\BaiduNetdiskDownload\ai_testing\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    corpus = load_corpus(corpus_path)
    vocab_size = tokenizer.vocab_size

    model = build_model(vocab_size, bert_path)
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y, attention_mask = build_dataset(batch_size, tokenizer, max_length, corpus)
            if torch.cuda.is_available():
                x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
            optim.zero_grad()
            loss = model(x, y, attention_mask)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.6f}")
        print(generate_sentence("罗伯斯干扰刘翔是否蓄谋已久？", model, tokenizer, max_length=50))
        print(generate_sentence("呼唤和培育新型官德", model, tokenizer, max_length=50))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace(".txt", ".pth")
        model_path = os.path.join("model", base_name)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train("sample_data.json", False)
