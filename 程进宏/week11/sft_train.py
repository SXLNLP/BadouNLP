# coding:utf8

import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

"""
基于pytorch的LSTM语言模型 -> bert+mark语言模型 -> bert+mark stf训练语言模型
1. 定义模型
2. 加载数据函数
3. 构建数据集函数
4. 模型建立函数
5. 文本生成测试代码
6. 采样策略函数
7. 主函数训练过程函数
8. 运行代码
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1) # 交叉熵损失，不计算 [CLS] + prompt + [SEP] 的损失，仅计算 answer + 结束标记[sep] 的损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            # mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            # print(mask, mask.shape)
            # if torch.cuda.is_available():
            #     mask = mask.cuda()

            # 传入一个 sft 训练的 mask 矩阵，问题可见，答案不可见
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

# 加载语料，读取json返回列表数据, title(prompt)，content(answer)
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus


# sft的数据构造
# loss只计算答案按部分，通过mask矩阵，让上下文之间没有交互
# label中使用-1，表示不参与训练
# 构建dataset列表【【x,mask,y】，【x,mask,y】】，传入Dataloander返回
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt,anwser) in enumerate(corpus):
        # 先读取 corpus 列表数据
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(anwser, add_special_tokens=False)
        # 注意是列表操作
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        # 构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = create_mask(len(prompt_encode), len(answer_encode))
        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    return DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=0)

# 构造掩码，输入两个字符串的长度
def create_mask(s1, s2):
    len_s1 = s1 + 2  # cls + sep
    len_s2 = s2 + 1  # sep
    # 创建掩码张量
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    # 遍历s1的每个token
    for i in range(len_s1):
        # s1的当前token不能看到s2的任何token
        mask[i, len_s1:] = 0
        # 遍历s2的每个token
    for i in range(len_s2):
        # s2的当前token不能看到后面的s2 token
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask

def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中（如果start不等与0其实是进行动态数据偏移将超出长度部分截断）
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
# def build_dataset(sample_length, tokenizer, window_size, corpus):
#     dataset_x = []
#     dataset_y = []
#     for i in range(sample_length):
#         x, y = build_sample(tokenizer, window_size, corpus)
#         dataset_x.append(x)
#         dataset_y.append(y)
#     return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码，bert编码解码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        # 生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)


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
    batch_size = 32  # 每次训练样本个数
    # train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    # window_size = 10  # 样本文本长度
    max_length = 50  # 样本文本长度
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r"D:\worksapce\ai_workspace\nlp20\week6\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料，返回列表
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  # 建立数据集，返回 DataLoader 对象
    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:  # 构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", True)
    # test();
