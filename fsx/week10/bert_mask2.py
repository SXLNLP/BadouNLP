# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from transformers import BertModel, BertTokenizer

"""
基于pytorch的自回归语言模型（使用上三角掩码）
"""
tokenizer = BertTokenizer.from_pretrained(
    '/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese')
bertModel = BertModel.from_pretrained('/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese',
                                      return_dict=False)


class LanguageModel(nn.Module):
    def __init__(self, input_dim=None):
        super(LanguageModel, self).__init__()
        self.model = bertModel
        # 修改分类器以适应自回归任务
        self.classify = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, y=None, causal_mask=None):
        # 使用BERT获取上下文表示，应用因果掩码
        if causal_mask is not None:
            # 将上三角掩码应用到注意力机制
            outputs = self.model(x, attention_mask=attention_mask, encoder_attention_mask=causal_mask)
        else:
            outputs = self.model(x, attention_mask=attention_mask)

        sequence_output = outputs[0]
        x = self.dropout(sequence_output)
        y_pred = self.classify(x)

        if y is not None:
            # 只计算预测下一个token的损失（错开一位）
            return self.loss(y_pred[:, :-1].contiguous().view(-1, y_pred.size(-1)),
                             y[:, 1:].contiguous().view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 生成上三角掩码（因果掩码）
def generate_causal_mask(seq_len):
    """生成上三角掩码矩阵，防止模型看到未来位置"""
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return mask


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 构建自回归样本
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]

    # 将字转换成序号
    x = [tokenizer.vocab.get(word, vocab["<UNK>"]) for word in window]

    # 创建attention mask（全部可见）
    attention_mask = [1] * window_size

    # 标签是输入序列向右移动一位（预测下一个token）
    y = x[1:] + [vocab["<UNK>"]]  # 最后一个token的标签不重要，设为<UNK>

    return x, attention_mask, y


# 建立数据集
def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_attn_mask = []
    dataset_y = []

    for i in range(sample_length):
        x, attn_mask, y = build_sample(tokenizer.vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_attn_mask.append(attn_mask)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_attn_mask), torch.LongTensor(dataset_y)


# 建立模型
def build_model(char_dim=None):
    model = LanguageModel(char_dim)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, max_length=30):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()

    generated_text = openings
    with torch.no_grad():
        # 生成文本直到遇到结束符或达到最大长度
        while len(generated_text) < max_length:
            # 准备输入
            x = [vocab.get(char, vocab["<UNK>"]) for char in generated_text]
            attention_mask = [1] * len(x)

            # 创建因果掩码
            causal_mask = generate_causal_mask(len(x))
            if torch.cuda.is_available():
                causal_mask = causal_mask.cuda()

            x_tensor = torch.LongTensor([x])
            attn_mask_tensor = torch.LongTensor([attention_mask])

            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                attn_mask_tensor = attn_mask_tensor.cuda()

            # 预测下一个字符
            y_pred = model(x_tensor, attn_mask_tensor, causal_mask=causal_mask)
            next_token_probs = y_pred[0, -1]  # 取最后一个token的预测

            # 采样策略
            if random.random() > 0.1:
                # 贪心策略
                next_token_id = int(torch.argmax(next_token_probs))
            else:
                # 随机采样
                probs = torch.softmax(next_token_probs, dim=0).cpu().numpy()
                next_token_id = np.random.choice(len(probs), p=probs)

            next_char = reverse_vocab.get(next_token_id, "[UNK]")

            # 如果生成了结束符，停止生成
            if next_char == "\n":
                break

            generated_text += next_char

    return generated_text


# 计算文本ppl
def calc_perplexity(sentence, model, vocab):
    log_prob_sum = 0
    model.eval()

    with torch.no_grad():
        for i in range(1, len(sentence)):
            # 准备上下文
            context = sentence[:i]
            target = sentence[i]

            # 转换为模型输入
            x = [vocab.get(char, vocab["<UNK>"]) for char in context]
            attention_mask = [1] * len(x)

            # 创建因果掩码
            causal_mask = generate_causal_mask(len(x))
            if torch.cuda.is_available():
                causal_mask = causal_mask.cuda()

            x_tensor = torch.LongTensor([x])
            attn_mask_tensor = torch.LongTensor([attention_mask])

            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                attn_mask_tensor = attn_mask_tensor.cuda()

            # 获取预测概率分布
            y_pred = model(x_tensor, attn_mask_tensor, causal_mask=causal_mask)
            target_id = vocab.get(target, vocab["<UNK>"])
            target_prob = torch.softmax(y_pred[0, -1], dim=0)[target_id].item()

            # 累加对数概率
            if target_prob > 0:
                log_prob_sum += math.log(target_prob)

    # 计算PPL
    if len(sentence) > 0:
        return math.exp(-log_prob_sum / (len(sentence) - 1))
    else:
        return float('inf')


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    window_size = 10  # 样本文本长度

    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model()  # 建立模型

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            # 构建一组训练样本
            x, attention_mask, y = build_dataset(batch_size, window_size, corpus)

            # 为每个样本创建因果掩码
            max_seq_len = x.size(1)
            causal_mask = generate_causal_mask(max_seq_len)

            if torch.cuda.is_available():
                x, attention_mask, y = x.cuda(), attention_mask.cuda(), y.cuda()
                causal_mask = causal_mask.cuda()

            optim.zero_grad()  # 梯度归零
            loss = model(x, attention_mask, y, causal_mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 生成文本示例
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer.vocab))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer.vocab))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    train("corpus.txt", False)
