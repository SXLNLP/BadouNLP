import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer, BertConfig, BertLMHeadModel


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, attention_mask=None, y=None):
        if isinstance(x, dict):
            bert_output = self.bert(**x)
        else:
            bert_output = self.bert(input_ids=x,
                                    attention_mask=attention_mask if attention_mask is not None
                                    else torch.ones_like(x))
        bert_output = self.dropout(bert_output)
        y_pred = self.classify(bert_output)
        if y is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = y_pred.view(-1, y_pred.shape[-1])[active_loss]
                active_labels = y.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


def build_bert_vocab(model_name="bert-base-chinese"):
    return BertTokenizer.from_pretrained(model_name)


def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]
    inputs = tokenizer(window, target, padding='max_length',
                       truncation=True, max_length=window_size,
                       return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']


def build_bert_dataset(sample_length, tokenizer, window_size, corpus):
    input_ids = []
    attention_masks = []
    for _ in range(sample_length):
        start = random.randint(0, len(corpus) - window_size - 1)
        text = corpus[start:start + window_size]
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=window_size,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return {
        'input_ids': torch.cat(input_ids),
        'attention_mask': torch.cat(attention_masks),
    }


def build_bert_model(vocab_size, hidden_dim=768):
    config = BertConfig(vocab_size=vocab_size,
                        hidden_size=hidden_dim,
                        num_hidden_layers=6,
                        num_attention_heads=8,
                        is_decoder=True)
    model = BertLMHeadModel(config)
    return model


def generate_sentence(prompt, model, tokenizer, max_length, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=30,
            top_p=0.85,
            temperature=1.0,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 64
    train_sample = 50000
    window_size = 128

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    config = BertConfig.from_pretrained("bert-base-chinese")
    config.is_decoder = True
    model = BertLMHeadModel.from_pretrained("bert-base-chinese", config=config)

    # 将模型移动到GPU
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.00001)
    corpus = load_corpus(corpus_path)

    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            inputs = build_bert_dataset(batch_size, tokenizer, window_size, corpus)

            # 将数据移动到GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            optim.zero_grad()
            loss = outputs[0]
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size, device))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size, device))

    if save_weight:
        model.save_pretrained("model/bert_finetuned")
        tokenizer.save_pretrained("model/bert_finetuned")


if __name__ == "__main__":
    train("corpus.txt", False)
