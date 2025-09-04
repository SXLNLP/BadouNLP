# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertTokenizer, BertModel


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy
        self.sep_token_id = self.bert.config.sep_token_id  # 获取BERT的[SEP] token id

    def forward(self, x, y=None):
        if y is not None:
            if not torch.is_tensor(x):
                x = torch.tensor(x, device='cuda' if torch.cuda.is_available() else 'cpu')

            batch_size, seq_len = x.shape
            mask = torch.zeros((batch_size, seq_len, seq_len), device=x.device)

            sep_positions = []
            for i in range(batch_size):
                sep_pos = seq_len - 1
                for j in range(seq_len):
                    if x[i, j].item() == self.sep_token_id:
                        sep_pos = j
                        break
                sep_positions.append(sep_pos)

            for i in range(batch_size):
                sep_pos = sep_positions[i]
                mask[i, :sep_pos + 1, :sep_pos + 1] = 1
                mask[i, sep_pos + 1:, :sep_pos + 1] = 1
                mask[i, sep_pos + 1:, sep_pos + 1:] = torch.tril(
                    torch.ones((seq_len - sep_pos - 1, seq_len - sep_pos - 1), device=x.device))

            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            if not torch.is_tensor(x):
                x = torch.tensor(x, device='cuda' if torch.cuda.is_available() else 'cpu')

            # 确保输入是二维的
            if x.dim() == 1:
                x = x.unsqueeze(0)

            seq_len = x.shape[1]

            # 创建下三角
            mask = torch.tril(torch.ones((1, seq_len, seq_len), device=x.device))

            # 找到分隔符位置
            sep_pos = seq_len - 1
            for j in range(seq_len):
                if x[0, j].item() == self.sep_token_id:
                    sep_pos = j
                    break

            # 允许title部分看到全部title内容
            mask[:, :sep_pos + 1, :sep_pos + 1] = 1

            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


def load_json_corpus(path):
    # JSON的读取需要多种处理，不确定是哪个
    with open(path, 'r', encoding='utf-8') as f:
        # 尝试读取为JSON数组
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError("Invalid JSON format")
        except json.JSONDecodeError:
            # 如果不是标准JSON，尝试逐行读取
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
            return data


def build_sample(tokenizer, max_length, data_item):
    title = data_item["title"]
    content = data_item["content"]

    # 添加明确的分隔符和指令
    text = f"生成新闻内容。标题：{title}。内容：{content}"
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 标签需要偏移1位（预测下一个token）
    labels = inputs["input_ids"].clone()
    labels[:, :-1] = labels[:, 1:].clone()  # 移位
    labels[:, -1] = -100  # 忽略填充部分的loss

    return inputs["input_ids"].squeeze(0), labels.squeeze(0)


def build_dataset(sample_length, tokenizer, max_length, corpus_data):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        # 随机选择一个数据项
        item = random.choice(corpus_data)
        x, y = build_sample(tokenizer, max_length, item)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.stack(dataset_y)


def generate_sentence(title, model, tokenizer, max_length):
    model.eval()
    with torch.no_grad():
        input_text = f"{title} [SEP]"
        input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        generated_ids = input_ids
        generated_text = ""

        # 逐步生成内容
        for _ in range(max_length - len(input_ids[0])):
            outputs = model(generated_ids)
            next_token_logits = outputs[0, -1, :]

            # 屏蔽[SEP]标记，避免过早终止
            next_token_logits[tokenizer.sep_token_id] = -float('inf')

            # 使用T采样
            next_token = temperature_sampling(next_token_logits, temperature=0.7)

            # 将新token添加到输入中
            generated_ids = torch.cat([
                generated_ids,
                torch.tensor([[next_token]], device=generated_ids.device)
            ], dim=1)
            generated_text += tokenizer.decode([next_token])

            # 遇到句号或换行符可以停止
            if next_token in {tokenizer.convert_tokens_to_ids('。'), tokenizer.convert_tokens_to_ids('\n')}:
                break

    return input_text + generated_text


def temperature_sampling(logits, temperature=1.0):
    # T采样策略
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = torch.softmax(prob_distribution, dim=-1).cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    max_length = 64  # 最大序列长度
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'D:\BaiduYunDownload\八斗精品班\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus_data = load_json_corpus(corpus_path)  # 加载JSON语料
    if not corpus_data:
        raise ValueError("No valid data loaded from JSON file")

    vocab_size = len(tokenizer)  # 获取tokenizer的词汇表大小

    model = LanguageModel(768, vocab_size, pretrain_model_path)  # 建立模型

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器

    print("模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, max_length, corpus_data)  # 构建训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 测试
        print(generate_sentence("阿根廷歹徒抢服装", model, tokenizer, max_length))
        print(generate_sentence("中国新闻网报道", model, tokenizer, max_length))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("sample_data.json", False)
