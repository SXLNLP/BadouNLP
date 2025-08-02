# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer
from config import Config
from loader import load_data
import logging

"""
基于pytorch的LSTM语言模型
"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        # attn_implementation 可以使mask 为3维传入
        self.bert = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False,
                                              attn_implementation='eager',
                                              num_hidden_layers=1)
        input_dim = self.bert.config.hidden_size
        self.classify = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        # self.loss = nn.functional.cross_entropy
        # 设置忽略标签值为 -100 的样本，使其不对损失计算产生影响。常用于处理变长序列中填充（padding）部分的标签
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # if torch.cuda.is_available():
            #     mask = mask.cuda()
            mask = self.build_mask(x)
            # 增加 batch_size 维度：(1, seq_len, seq_len)
            mask = mask.unsqueeze(0)  # shape: (1, 5, 5)
            # 扩展为 (batch_size, seq_len, seq_len)
            mask = mask.expand(x.shape[0], -1, -1)  # shape: (4, 5, 5)
            x, _ = self.bert(x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim) 64 10 256
            x = self.dropout(x)  # 32 150 768
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size) 64 10 3961
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时 可以不使用mask
            x, _ = self.bert(x)
            x = self.dropout(x)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size) 64 10 3961
            return torch.softmax(y_pred, dim=-1)

    def build_mask(self, x):
        x_len = x.shape[1]
        # 查找 102 的位置
        sep_positions = (x == 102).nonzero(as_tuple=True)[1]
        # print(sep_positions)  # 输出: tensor([3, 5])
        mask = torch.zeros(x_len, x_len, dtype=torch.float32)
        # print(mask)
        for i in range(x_len):
            for j in range(x_len):
                if i < sep_positions[0] and j < sep_positions[0]:
                    mask[i][j] = 1  # 前 sep_index 行列全为 1
                elif j <= i:
                    mask[i][j] = 1  # 后续部分为下三角矩阵
        # print(mask)
        return mask

    def forward1(self, input_seq, output_seq=None, y=None):
        x = input_seq
        if y is not None:
            # 32 120      32 30
            x = torch.cat((input_seq, output_seq), dim=1)
            input_seq_len = input_seq.shape[1]  # 120
            output_seq_len = output_seq.shape[1]  # 30
            left1_mask = torch.ones(input_seq_len, input_seq_len, dtype=torch.float32)
            # print(left1_mask)
            right1_mask = torch.zeros(input_seq_len, output_seq_len, dtype=torch.float32)
            # print(right1_mask)
            left2_mask = torch.ones(output_seq_len, input_seq_len, dtype=torch.float32)
            # print(left2_mask)
            right2_mask = torch.tril(torch.ones(output_seq_len, output_seq_len, dtype=torch.float32))
            # print(right2_mask)
            concatenated_mask1 = torch.cat((left1_mask, right1_mask), dim=1)
            concatenated_mask2 = torch.cat((left2_mask, right2_mask), dim=1)
            concatenated_mask = torch.cat((concatenated_mask1, concatenated_mask2), dim=0)
            # 增加 batch_size 维度：(1, seq_len, seq_len)
            mask = concatenated_mask.unsqueeze(0)  # shape: (1, 5, 5)
            # 扩展为 (batch_size, seq_len, seq_len)
            mask = mask.expand(input_seq.shape[0], -1, -1)  # shape: (4, 5, 5)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim) 64 10 256
            x = self.dropout(x)  # 32 150 768
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size) 64 10 3961
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时 可以不使用mask
            x, _ = self.bert(x)
            x = self.dropout(x)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size) 64 10 3961
            return torch.softmax(y_pred, dim=-1)


# 建立模型
def build_model(vocab_size):
    model = LanguageModel(vocab_size)
    return model


# 文本生成测试代码
# def generate_sentence(openings, model, tokenizer):
#     model.eval()
#     with torch.no_grad():
#         pred_char = ""
#         sep_token = tokenizer.decode(102)
#         # 生成了换行符，或生成文本超过30字则终止迭代
#         while pred_char != "\n" and len(openings) <= 80 and pred_char != sep_token:
#             openings += pred_char
#             x = tokenizer.encode(openings, add_special_tokens=False)
#             x = torch.LongTensor([x])
#             if torch.cuda.is_available():
#                 x = x.cuda()
#             y = model(x)[0][-1]
#             index = sampling_strategy(y)
#             pred_char = ''.join(tokenizer.decode(index))
#     return openings


def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
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


#
# def sampling_strategy(logits, repetition_penalty=1.2, temperature=0.7, top_k=50, top_p=0.9):
#     # 清除缓存状态（generate_sentence 中调用）
#     if not hasattr(sampling_strategy, "history_tokens"):
#         sampling_strategy.history_tokens = []
#
#     # 应用重复惩罚
#     for token_id in set(sampling_strategy.history_tokens):
#         logits[token_id] /= repetition_penalty
#
#     # softmax + 温度缩放
#     probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
#
#     # Top-k 采样
#     indices = np.argsort(probs)[-top_k:]
#     top_probs = probs[indices]
#
#     # Nucleus (Top-p) 采样
#     sorted_indices = np.argsort(top_probs)[::-1]
#     sorted_probs = top_probs[sorted_indices]
#     cumulative_probs = np.cumsum(sorted_probs)
#     sorted_probs[cumulative_probs > top_p] = 0
#     sorted_probs /= sorted_probs.sum()  # 归一化
#
#     # 随机选择
#     next_token_idx = np.random.choice(len(sorted_probs), p=sorted_probs)
#     next_token = indices[sorted_indices[next_token_idx]]
#
#     # 记录已生成 token
#     sampling_strategy.history_tokens.append(next_token)
#
#     return int(next_token)


def train(corpus_path, save_weight=True):
    epoch_num = 100  # 训练轮数
    tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
    train_data = load_data(corpus_path, Config, logger)
    model = build_model(len(tokenizer.vocab))  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # for batch in range(int(train_sample / batch_size)):
        #     x, y = build_dataset(batch_size, tokenizer, window_size, corpus)  # 构建一组训练样本
        for batch_data in train_data:
            # input_seq, output_seq, gold = batch_data
            # if torch.cuda.is_available():
            #     input_seq, output_seq, gold = input_seq.cuda(), output_seq.cuda(), gold.cuda()
            input_seq, gold = batch_data
            if torch.cuda.is_available():
                input_seq, gold = input_seq.cuda(), gold.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(input_seq, gold)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence(
            "地球公转周期是多久",
            model, tokenizer))
        print(generate_sentence(
            "水由什么元素组成",
            model, tokenizer))
        print(generate_sentence(
            "光速是多少",
            model, tokenizer))
        print(generate_sentence(
            "什么是Python",
            model, tokenizer))
        print(generate_sentence(
            "哪个行星是红色的",
            model, tokenizer))
        print(generate_sentence(
            "谁创作了《哈姆雷特》",
            model, tokenizer))
    if not save_weight:
        return
    else:
        model_path = os.path.join(Config["model_path"], "epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("sample_data.json", True)

    # # 假设 x 是一个 tensor
    # x = torch.tensor([[101,999, 123, 456, 102, 789, 756, 888]])
    # x_len = x.shape[1]
    # # 查找 102 的位置
    # sep_positions = (x == 102).nonzero(as_tuple=True)[1]
    # print(sep_positions)  # 输出: tensor([3, 5])
    # mask = torch.zeros(x_len, x_len, dtype=torch.float32)
    # # print(mask)
    # for i in range(x_len):
    #     for j in range(x_len):
    #         if i < sep_positions[0] and j < sep_positions[0]:
    #             mask[i][j] = 1  # 前 sep_index 行列全为 1
    #         elif j <= i:
    #             mask[i][j] = 1  # 后续部分为下三角矩阵
    # print(mask)

    # tensor([[1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1.]])
