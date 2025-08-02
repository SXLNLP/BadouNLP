# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from transformers import BertTokenizer, BertModel

"""
基于BERT的标题生成模型(SFT)
将新闻内容作为输入，生成对应的新闻标题
"""

class BertLanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(BertLanguageModel, self).__init__()
        # 使用预训练的BERT模型作为编码器
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        # 分类层，用于预测下一个token
        self.classify = nn.Linear(hidden_size, vocab_size)
        # 损失函数
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，实现因果注意力机制
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append({
                    'content': item['content'],  # 内容作为输入
                    'title': item['title']       # 标题作为输出目标
                })
    return data

def build_sample(tokenizer, data_item, max_length=128):
    content = data_item['content']  # 现在内容是输入
    title = data_item['title']      # 现在标题是目标

    # 编码内容部分（输入）
    content_tokens = tokenizer.encode(
        content,
        add_special_tokens=True,  # 添加[CLS]和[SEP]
        truncation=True,
        max_length=max_length//2
    )

    # 编码标题部分（目标）
    title_tokens = tokenizer.encode(
        title,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length//2
    )

    # 组合序列：[CLS] 内容 [SEP] 标题 [SEP]
    input_tokens = content_tokens + title_tokens + [tokenizer.sep_token_id]

    # 构建标签序列 - 内容部分用0填充(不计算损失)，标题部分保留真实token
    labels = [0] * len(content_tokens) + title_tokens + [tokenizer.sep_token_id]
    if len(labels) > max_length:
        labels = labels[:max_length]

    # 如果序列长度不足，进行padding
    while len(input_tokens) < max_length:
        input_tokens.append(tokenizer.pad_token_id)

    while len(labels) < max_length:
        labels.append(0)  # 0作为忽略的标签ID

    return input_tokens, labels

# 建立数据集
def build_dataset(sample_length, tokenizer, data, max_length=128):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        # 随机选择一个数据项
        item = random.choice(data)
        x, y = build_sample(tokenizer, item, max_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(pretrain_model_path):
    model = BertLanguageModel(768, 21128, pretrain_model_path)
    return model

# 根据内容生成标题
def generate_title(content, model, tokenizer, max_length=128, max_title_length=40):
    model.eval()
    with torch.no_grad():
        # 编码内容输入
        content_tokens = tokenizer.encode(
            content,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            max_length=max_length//2
        )

        if torch.cuda.is_available():
            content_tokens = content_tokens.cuda()

        # 记录原始内容长度，用于后续分离标题部分
        content_length = content_tokens.shape[1]

        # 逐个生成标题token
        for _ in range(max_title_length):
            # 获取模型预测
            y_pred = model(content_tokens)
            
            # 获取最后一个token的预测
            next_token_logits = y_pred[0][-1]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # 如果生成结束标记则停止
            if next_token_id == tokenizer.sep_token_id:
                break
                
            # 将预测的token添加到序列中
            next_token_tensor = torch.tensor([[next_token_id]], device=content_tokens.device)
            content_tokens = torch.cat([content_tokens, next_token_tensor], dim=1)
            
            # 防止超过最大长度
            if content_tokens.shape[1] >= max_length:
                break

        # 解码生成的标题部分（去除内容部分和特殊标记）
        generated_tokens = content_tokens[0][content_length:].tolist()
        
        # 移除可能的pad token和sep token
        title_tokens = []
        for token_id in generated_tokens:
            if token_id == tokenizer.sep_token_id:
                break
            if token_id != tokenizer.pad_token_id:
                title_tokens.append(token_id)
        
        # 解码为文本
        title = tokenizer.decode(title_tokens, skip_special_tokens=True)
        return title

def train(data_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 32       # 每次训练样本个数
    train_sample = 5000   # 每轮训练总共训练的样本总数
    max_length = 128      # 序列最大长度
    learning_rate = 0.0001  # 学习率

    # 预训练模型路径
    pretrain_model_path = r'D:\nlp516\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # 加载数据
    data = load_data(data_path)
    print(f"加载了 {len(data)} 条数据")

    # 建立模型
    model = build_model(pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    # 建立优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # 构建一批训练样本
            x, y = build_dataset(batch_size, tokenizer, data, max_length)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss)}")

        # 测试生成效果
        test_items = random.sample(data, 3)  # 随机选择3个测试样本
        for item in test_items:
            content = item['content']     # 现在用内容作为输入
            true_title = item['title']    # 真实标题
            generated_title = generate_title(content, model, tokenizer, max_length)  # 生成标题
            print(f"输入内容: {content[:50]}...")  # 只显示前50个字符
            print(f"真实标题: {true_title}")
            print(f"生成标题: {generated_title}\n")

    if save_weight:
        model_path = os.path.join("model", "title_model.pth")  # 修改保存文件名
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")

if __name__ == "__main__":
    train("sample_data.json", False)