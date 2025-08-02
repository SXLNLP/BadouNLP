# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from transformers import BertTokenizer, BertModel

"""
用bert作为基座模型，修改mask，实现sft
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        batch_size, seq_len = x.shape
        # Title部分不遮挡，Content部分使用下三角遮挡
        mask = torch.ones((batch_size, seq_len, seq_len))  # 全1矩阵

        for i in range(batch_size):
            mask[i, :, :] = torch.tril(mask[i, :, :])  # 只保留下三角

        if y is not None:
            # Training mode: 设置mask，仅遮挡Content部分
            mask = mask.cuda() if torch.cuda.is_available() else mask
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # Prediction mode: 不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


def load_json_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                if line.strip():  # 确保不是空行
                    data.append(json.loads(line.strip()))  # 解析每一行
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")  # 打印解析失败的行
        print(f"Loaded {len(data)} samples.")  # 确认加载的样本数量
    return data


# title和content分别作为问题和答案
def build_sample(tokenizer, window_size, title, content):
    # 这里title作为问题，content作为答案
    combined = title + ' ' + content
    start = random.randint(0, len(combined) - window_size)
    end = start + window_size
    window = combined[start:end]
    target = combined[start + 1:end + 1]  # 输入输出错开一位

    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True,
                         max_length=window_size)
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True,
                         max_length=window_size)

    return x, y


def build_dataset(sample_length, tokenizer, window_size, data):
    dataset_x = []
    dataset_y = []
    total_samples = len(data)

    # 确保样本长度不超过数据的总长度
    sample_length = min(sample_length, total_samples)
    print(f"Warning: sample_length adjusted to {sample_length} based on data size.")

    for i in range(sample_length):
        title = data[i]['title']
        content = data[i]['content']
        x, y = build_sample(tokenizer, window_size, title, content)
        dataset_x.append(x)
        dataset_y.append(y)

    print(f"Generated {len(dataset_x)} samples.")
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)



# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_answer(title, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_answer = title  # 初始是问题部分
        x = tokenizer.encode(title, add_special_tokens=True, truncation=True, max_length=window_size)
        x = torch.LongTensor([x])
        if torch.cuda.is_available():
            x = x.cuda()

        # 生成答案部分，直到生成结束符或超过30个词
        while len(pred_answer.split()) < 30:  # 生成30个词以内的答案
            y = model(x)  # 获取预测结果
            pred_token = torch.argmax(y, dim=-1)[:, -1]  # 取最后一个token的预测
            pred_char = tokenizer.decode(pred_token.cpu().numpy())

            if pred_char == "<|endoftext|>":  # 假设BERT的结束标记是"<|endoftext|>"
                break

            pred_answer += pred_char  # 拼接预测的字符
            x = torch.cat([x, pred_token.unsqueeze(1)], dim=1)  # 将新的预测token加入输入

    return pred_answer


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


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'/Users/ge/PycharmProjects/pythonProject/bd/week10/homework/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    data = load_json_corpus(corpus_path)  # 加载JSON数据
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, data)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 在每轮训练后生成问题的答案
        print(generate_answer("南宁否认限购松绑系救市：实施北部湾城镇体系同城化", model, tokenizer, window_size))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"/Users/ge/PycharmProjects/pythonProject/bd/week11/sample_data.json", False)
