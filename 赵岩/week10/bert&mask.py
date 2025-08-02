# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
from torch.optim import AdamW  # 使用PyTorch自带的AdamW
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler  # 导入自动混合精度工具

"""
基于BERT的自回归语言模型
"""

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.examples = []
        with open(file_path, encoding="gbk") as f:  # 指定正确的编码格式
            text = f.read()

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size])
            )
        print(f"加载了 {len(self.examples)} 个示例")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            if not line.startswith("本书由新奇书网"):
                corpus += line.strip()
    return corpus


# 建立模型
def build_model():
    local_model_path = 'E:/BaiduNetdiskDownload/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertForMaskedLM.from_pretrained(local_model_path)

    # 冻结部分层以减少显存占用
    for param in model.bert.encoder.layer[:6].parameters():
        param.requires_grad = False

    return model, tokenizer


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, max_length=50, top_k=50, temperature=1.0, num_beams=5):
    reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(openings, add_special_tokens=False)
        padding_length = 512 - len(input_ids)  # BERT的最大序列长度为512
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * len(input_ids)

        x = torch.LongTensor([input_ids]).to(device)
        attention_mask = torch.LongTensor([attention_mask]).to(device)

        generated_sequence = input_ids.copy()

        while len(generated_sequence) < max_length and '[SEP]' not in tokenizer.decode(generated_sequence):
            outputs = model(input_ids=x, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            last_token_logits = logits[0, -1, :] / temperature

            filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=None)
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
            index = torch.multinomial(probs, num_samples=1).item()
            generated_sequence.append(index)

            # 更新 input_ids 和 attention_mask
            input_ids = generated_sequence[-512:]  # 只保留最后512个token
            padding_length = 512 - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids)

            x = torch.LongTensor([input_ids]).to(device)
            attention_mask = torch.LongTensor([attention_mask]).to(device)

            # 调试信息
            print(f"生成的Token: {reverse_vocab.get(index, '[UNK]')}")

    return tokenizer.decode(generated_sequence).split('[SEP]')[0]


def beam_search_generate_sentence(openings, model, tokenizer, max_length=50, top_k=50, temperature=1.0, num_beams=5):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(openings, add_special_tokens=False)
        padding_length = 512 - len(input_ids)  # BERT的最大序列长度为512
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * len(input_ids)

        x = torch.LongTensor([input_ids]).to(device)
        attention_mask = torch.LongTensor([attention_mask]).to(device)

        beams = [(x, attention_mask, [])]  # (当前输入ids, 当前注意力掩码, 生成序列)

        for _ in range(max_length):
            new_beams = []
            for beam in beams:
                current_input_ids, current_attention_mask, generated_sequence = beam
                outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                last_token_logits = logits[0, -1, :] / temperature

                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=None)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, num_beams)

                for prob, index in zip(top_probs, top_indices):
                    new_generated_sequence = generated_sequence + [index.item()]
                    new_input_ids = new_generated_sequence[-512:]  # 只保留最后512个token
                    padding_length = 512 - len(new_input_ids)
                    new_input_ids = new_input_ids + [tokenizer.pad_token_id] * padding_length
                    new_attention_mask = [1] * len(new_input_ids)

                    new_x = torch.LongTensor([new_input_ids]).to(device)
                    new_attention_mask = torch.LongTensor([new_attention_mask]).to(device)

                    new_beams.append((new_x, new_attention_mask, new_generated_sequence))

            # 按概率排序并保留最佳num_beams条路径
            new_beams.sort(key=lambda x: sum(x[2]), reverse=True)
            beams = new_beams[:num_beams]

        最佳路径 = beams[0]
        最终序列 = 最佳路径[2]
        return tokenizer.decode(最终序列).split('[SEP]')[0]


def top_k_top_p_filtering(logits, top_k=50, top_p=None, filter_value=-float('Inf')):
    """ 使用top-k和/或核采样（top-p）过滤logits分布
        参数:
            logits: logits 分布形状 (batch size, vocabulary size)
            top_k >0: 仅保留最高概率的top k个token (top-k 过滤).
            top_p >0.0: 保留累积概率 >= top_p 的顶级token (核采样).
    """
    assert logits.dim() == 1  # 目前仅支持批处理大小为1的情况 - 后续可以扩展
    top_k = min(top_k, logits.size(-1))  # 安全检查
    if top_k > 0:
        # 移除所有概率低于top-k最后一个token的所有token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p is not None and top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率高于阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引右移以保持阈值以上的第一个token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


# 计算文本困惑度
def calc_perplexity(sentence, model, tokenizer, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            target = sentence[i]
            input_ids = tokenizer.encode(window, add_special_tokens=False)
            padding_length = window_size - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids)

            x = torch.LongTensor([input_ids]).to(device)
            attention_mask = torch.LongTensor([attention_mask]).to(device)

            outputs = model(input_ids=x, attention_mask=attention_mask)
            logits = outputs.logits
            last_token_logits = logits[0, -1, :]
            probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
            target_index = tokenizer.convert_tokens_to_ids(target)
            target_prob = probs[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 16       # 每次训练样本个数
    train_sample = 50000   # 每轮训练总共训练的样本总数
    char_dim = 256        # 每个字符的维度
    window_size = 128       # 样本文本长度
    vocab = build_vocab("vocab.txt")       # 建立字表
    corpus = load_corpus(corpus_path)     # 加载语料
    model, tokenizer = build_model()    # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = AdamW(model.parameters(), lr=1e-5)   # 建立优化器
    total_steps = epoch_num * (train_sample // batch_size)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)

    scaler = GradScaler('cuda')  # 初始化GradScaler用于自动混合精度

    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus) # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    # 梯度归零
            with autocast('cuda'):
                loss = model(input_ids=x, labels=y).loss  # 计算损失
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optim)  # 更新权重
            scaler.update()  # 更新scaler
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


# 构建数据集
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    x = tokenizer.encode(window, add_special_tokens=False)   # 将字符转换成id
    y = tokenizer.encode(target, add_special_tokens=False)
    return x, y


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       # 去掉结尾换行符
            vocab[char] = index + 1 # 留出0位给pad token
    return vocab


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train("corpus.txt", False)



