#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

"""
基于pytorch和transformers库的BERT掩码语言模型
"""


class BERTLanguageModel(nn.Module):
    def __init__(self, vocab_size, model_name='bert-base-chinese'):
        super(BERTLanguageModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)
        self.model = BertForMaskedLM.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)
        self.vocab_size = vocab_size
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        return outputs

# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1  # 留出0位给pad token
    # 添加特殊标记
    vocab["[MASK]"] = len(vocab)
    vocab["[CLS]"] = len(vocab)
    vocab["[SEP]"] = len(vocab)
    vocab["[UNK]"] = 1  # 保持与原代码一致
    return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk", errors='ignore') as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
# 从文本中截取随机窗口，并随机mask一些token
def build_sample(vocab, window_size, corpus, tokenizer, mask_prob=0.15):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]

    # 使用BERT tokenizer进行tokenize
    tokens = tokenizer.tokenize(window)
    # 限制长度并添加特殊标记
    tokens = tokens[:window_size-2]  # 留出空间给[CLS]和[SEP]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    # 转换为ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 创建标签，初始值为-100（损失函数会忽略这个值）
    labels = [-100] * len(input_ids)

    # 随机选择一些位置进行mask
    for i in range(1, len(input_ids)-1):  # 不mask [CLS] 和 [SEP]
        if random.random() < mask_prob:
            labels[i] = input_ids[i]  # 保存原始值用于计算损失
            # 80%的概率替换为[MASK]
            if random.random() < 0.8:
                input_ids[i] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
            # 10%的概率替换为随机token
            elif random.random() < 0.5:  # 0.5 * 0.2 = 0.1
                input_ids[i] = random.randint(0, len(tokenizer.vocab) - 1)
            # 10%的概率保持不变（这里不需要操作）

    return input_ids, labels

# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus, tokenizer):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus, tokenizer)
        dataset_x.append(x)
        dataset_y.append(y)

    # 对数据进行padding，使所有序列长度一致
    max_len = max(len(x) for x in dataset_x)
    padded_x = []
    padded_y = []
    for x, y in zip(dataset_x, dataset_y):
        pad_len = max_len - len(x)
        padded_x.append(x + [0] * pad_len)  # 用0进行padding
        padded_y.append(y + [-100] * pad_len)  # 用-100进行padding

    return torch.LongTensor(padded_x), torch.LongTensor(padded_y)

# 建立模型
def build_model(vocab_size, model_name='bert-base-chinese'):
    model = BERTLanguageModel(vocab_size, model_name)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size, max_len=30):
    model.eval()
    with torch.no_grad():
        # 将输入文本转换为token ids
        input_ids = tokenizer.encode(openings, add_special_tokens=True)

        # 逐步生成文本
        for _ in range(max_len - len(input_ids)):
            input_tensor = torch.tensor([input_ids])
            attention_mask = torch.ones_like(input_tensor)

            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                attention_mask = attention_mask.cuda()
                model = model.cuda()

            # 获取模型预测
            outputs = model(input_tensor, attention_mask=attention_mask)

            # 修复输出处理逻辑
            if isinstance(outputs, tuple):
                # 如果是元组类型，直接取第一个元素（logits）
                logits = outputs[0]
            else:
                # 如果是对象类型，取logits属性
                logits = outputs.logits

            # 获取最后一个token的预测结果
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()

            # 如果是结束符，则停止
            if next_token_id == tokenizer.sep_token_id:
                break

            # 添加预测的token到输入序列中
            input_ids.append(next_token_id)

            # 将生成的token id转换为文本
            generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            if len(generated_text) > len(openings) and generated_text.endswith('\n'):
                break

    return tokenizer.decode(input_ids, skip_special_tokens=True)

# 计算文本ppl
def calc_perplexity(sentence, model, tokenizer):
    model.eval()
    with torch.no_grad():
        # 编码句子
        inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            model = model.cuda()

        # 计算损失
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

        # 修复输出处理逻辑
        if isinstance(outputs, tuple):
            loss = outputs[0]  # 第一个元素是loss
        else:
            loss = outputs.loss

        # 困惑度是损失的指数
        ppl = torch.exp(loss)
    return ppl.item()

def train(corpus_path, save_weight=True):
    epoch_num = 5         # 训练轮数
    batch_size = 16       # 每次训练样本个数（减少batch size以适应显存）
    train_sample = 1000   # 每轮训练总共训练的样本总数（减少样本数以加快训练）
    window_size = 32      # 样本文本长度

    # 构建完整的文件路径
    base_dir = os.path.dirname(corpus_path)
    vocab_path = os.path.join(base_dir, "vocab.txt")

    # 检查文件是否存在
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")

    vocab = build_vocab(vocab_path)  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    tokenizer = BertTokenizer.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)

    # 建立模型
    model = build_model(len(vocab))
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=5e-5)  # BERT推荐学习率
    print("BERT语言模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus, tokenizer)  # 构建一组训练样本
            attention_mask = (x != 0).long()  # 创建attention mask
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
                attention_mask = attention_mask.cuda()

            optim.zero_grad()    # 梯度归零
            outputs = model(x, attention_mask=attention_mask, labels=y)   # 计算loss
            if isinstance(outputs, tuple):
                loss = outputs[0]  # 第一个元素是loss
            else:
              loss = outputs.loss            
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 生成句子示例
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", "bert_" + base_name)
        # 确保模型保存目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # 使用完整路径
    current_dir = "F:/BaiduNetdiskDownload/八斗精品班/第十周/week10 文本生成问题/lstm语言模型生成文本/"
    corpus_path = os.path.join(current_dir, "corpus.txt")
    train(corpus_path, False)
