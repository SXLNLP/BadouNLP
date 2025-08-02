#coding:utf8

import torch
import torch.nn as nn
import random
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np


"""
基于 HuggingFace Transformers 的 BERT 模型训练语言模型
"""

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(r"D:\nlp516\bert-base-chinese")

# 模型定义
from transformers import BertConfig

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        # 1. 加载配置并设置为 decoder 模式
        config = BertConfig.from_pretrained(r"D:\nlp516\bert-base-chinese")
        config.is_decoder = True  # 设置为 decoder 模式，单向模式
        config.add_cross_attention = False

        # 2. 用配置初始化 BERT 模型
        self.bert = BertModel(config)
        self.vocab_size = tokenizer.vocab_size
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if labels is not None:
            # ✅ 预测第 1~15 个 token 的下一个
            # ✅ 真实标签是第 2~16 个 token
            shift_logits = logits[..., :-1, :].contiguous()   # [batch, 15, vocab]
            shift_labels = labels[..., 1:].contiguous()       # [batch, 15]

            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),  # [batch*15, vocab]
                shift_labels.view(-1)                          # [batch*15]
            )
            return loss, logits
        else:
            return logits
        
# .contiguous() 用于 确保张量在内存中是连续存储的。

def build_sample(window):
    encoding = tokenizer(
        window,
        max_length=16,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].squeeze(0)      # [16]
    attention_mask = encoding['attention_mask'].squeeze(0)  # [16]

    # input_ids: 第 1~16 个 token
    # labels: 和 input_ids 一样，交给模型处理 shift
    labels = input_ids.clone()

    return input_ids, attention_mask, labels


# 构建数据集
class LanguageDataset(Dataset):
    def __init__(self, corpus, window_size=16, sample_length=10000):
        self.corpus = corpus
        self.window_size = window_size
        self.sample_length = sample_length

    def __len__(self):
        return self.sample_length

    def __getitem__(self, idx):
        start = random.randint(0, len(self.corpus) - self.window_size - 1)
        window = self.corpus[start:start + self.window_size]
        return build_sample(window)

# 文本生成函数
def generate_sentence(prompt, model, max_len=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    generated_text = prompt  # 用于记录生成的完整文本

    for _ in range(max_len - len(input_ids[0])):
        with torch.no_grad():
            logits = model(input_ids, attention_mask=(input_ids != tokenizer.pad_token_id).long())
            next_token_logits = logits[:, -1, :]  # 取最后一个 token 的 logits
            # 使用采样策略
            next_token_id = sampling_strategy(next_token_logits)
            # 解码 token
            next_char = tokenizer.decode([next_token_id])

            # 判断是否遇到换行符或长度超过36
            if next_char == '\n' or len(generated_text) >= 36:
                break
            # 更新生成内容
            generated_text += next_char
            next_token = torch.tensor([[next_token_id]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return generated_text

def sampling_strategy(prob_distribution):
    prob_distribution = torch.softmax(prob_distribution, dim=-1)  # 归一化为概率分布
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy().squeeze()
        return int(np.random.choice(
            list(range(len(prob_distribution))),
            p=prob_distribution
        ))


# 训练函数
def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数 64
    train_sample = 10000   #每轮训练总共训练的样本总数  50000
    window_size = 16       #样本文本长度
    
    # 加载语料
    corpus = ""
    with open(corpus_path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    print("语料加载完成！")

    # 构建数据集
    dataset = LanguageDataset(corpus, window_size=window_size, sample_length=train_sample)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = LanguageModel()
    if torch.cuda.is_available():
        model = model.cuda()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # 训练循环
    print("使用BERT语言模型+attention_mask开始自回归语言模型（encoder-decoder）训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            if torch.cuda.is_available():
                input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(dataloader):.4f}")
        print(generate_sentence("让他在半年之前，就不能做出", model))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model))

    if save_weight:
        torch.save(model.state_dict(), "bert_language_model.pth")


if __name__ == "__main__":
    train("corpus.txt",True)

#     Epoch 20, Avg Loss: 4.0593
# 让他在半年之前，就不能做出[SEP]，他们，这些人，他们都没那么了，他们
# 李慕站在山路上，深深的呼吸[SEP]了一个人，他们，他们的时间内，说道：