# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import json
from sklearn.model_selection import train_test_split
from transformers import BertModel

"""
基于pytorch的SFT语言模型，支持验证集和标题到内容生成测试
S1为title，S2为content
"""


def get_s1_s2_mask(seq, s1_lengths):
    '''
    生成S1和S2的特殊掩码:
    - S1只能看到自身全部内容（不能看S2）
    - S2可以看到S1全部内容和自身之前的内容，看不到自身之后的内容
    '''
    batch_size, seq_len = seq.size()
    mask = torch.ones((batch_size, seq_len, seq_len), device=seq.device)

    for i in range(batch_size):
        s1_len = s1_lengths[i]

        # 1. 限制S1只能关注自身（不能关注S2）
        # S1的行（0到s1_len-1）只能关注S1的列（0到s1_len-1）
        mask[i, :s1_len, s1_len:] = 0  # S1的行 → S2的列：设为0（不可见）

        # 2. S2部分使用下三角掩码（只能看到自身之前的内容）
        if seq_len - s1_len > 0:
            s2_mask = (1 - torch.triu(
                torch.ones((seq_len - s1_len, seq_len - s1_len), device=seq.device),
                diagonal=1
            )).bool()
            mask[i, s1_len:, s1_len:] = s2_mask

        # 3. S2可以看到所有S1内容
        mask[i, s1_len:, :s1_len] = 1

    return mask.bool()


class SFTLanguageModel(nn.Module):
    def __init__(self, vocab):
        super(SFTLanguageModel, self).__init__()
        self.vocab = vocab
        self.bert = BertModel.from_pretrained(r'E:\八斗精品班\第六周 语言模型\bert-base-chinese\bert-base-chinese')
        self.classify = nn.Linear(self.bert.config.hidden_size, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, s1_lengths, y=None):
        # 生成特殊掩码
        mask = get_s1_s2_mask(x, s1_lengths)

        # 通过BERT编码
        outputs = self.bert(input_ids=x, attention_mask=mask)
        hidden_states = self.dropout(outputs[0])
        y_pred = self.classify(hidden_states)  # [batch_size, seq_len, vocab_size]

        if y is not None:
            # 忽略padding位置的损失
            pad_mask = (y != self.vocab["<pad>"]).float().view(-1)
            loss = self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), reduction='none')
            return (loss * pad_mask).sum() / pad_mask.sum()
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0, "<UNK>": 1, "<SEP>": 2, "<EOS>": 3}  # 增加结束符
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 4  # 留出前4位给特殊符号
    return vocab


# 加载JSON格式的SFT语料
def load_json_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    s1 = data.get("title", "").strip()
                    s2 = data.get("content", "").strip()
                    if s1 and s2:
                        corpus.append((s1, s2))
                except json.JSONDecodeError:
                    print(f"警告：无法解析JSON行 - {line}")
    return corpus


# 分割训练集和验证集
def split_train_val(corpus, val_ratio=0.1):
    return train_test_split(corpus, test_size=val_ratio, random_state=42)


# 构建样本
def build_sft_sample(vocab, max_seq_len, s1, s2):
    s1_ids = [vocab.get(c, vocab["<UNK>"]) for c in s1]
    s2_ids = [vocab.get(c, vocab["<UNK>"]) for c in s2] + [vocab["<EOS>"]]  # 添加结束符
    sep_id = vocab["<SEP>"]

    # 组合序列
    input_ids = s1_ids + [sep_id] + s2_ids
    s1_len = len(s1_ids) + 1  # 包含分隔符

    # 标签是输入的偏移
    labels = input_ids[1:] + [vocab["<pad>"]]

    # 截断过长序列
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        s1_len = min(s1_len, max_seq_len)

    # 填充到最大长度
    pad_len = max_seq_len - len(input_ids)
    input_ids += [vocab["<pad>"]] * pad_len
    labels += [vocab["<pad>"]] * pad_len

    return input_ids, labels, s1_len


# 构建数据集
def build_sft_dataset(samples, vocab, max_seq_len, corpus):
    dataset_x = []
    dataset_y = []
    dataset_s1_len = []

    for _ in range(samples):
        s1, s2 = random.choice(corpus)
        x, y, s1_len = build_sft_sample(vocab, max_seq_len, s1, s2)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_s1_len.append(s1_len)

    return (torch.LongTensor(dataset_x),
            torch.LongTensor(dataset_y),
            torch.LongTensor(dataset_s1_len))


# 构建验证集（不重复采样）
def build_validation_dataset(vocab, max_seq_len, val_corpus):
    dataset_x = []
    dataset_y = []
    dataset_s1_len = []

    for s1, s2 in val_corpus:
        x, y, s1_len = build_sft_sample(vocab, max_seq_len, s1, s2)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_s1_len.append(s1_len)

    return (torch.LongTensor(dataset_x),
            torch.LongTensor(dataset_y),
            torch.LongTensor(dataset_s1_len))


# 生成内容：输入标题返回内容
def generate_content_by_title(title, model, vocab, max_seq_len, temperature=1.0):
    reverse_vocab = {v: k for k, v in vocab.items()}
    model.eval()
    with torch.no_grad():
        # 处理标题部分
        s1_ids = [vocab.get(c, vocab["<UNK>"]) for c in title] + [vocab["<SEP>"]]
        s1_len = len(s1_ids)
        current_ids = s1_ids.copy()

        # 生成内容直到遇到结束符或达到最大长度
        while len(current_ids) < max_seq_len:
            # 准备输入
            input_ids = current_ids[-max_seq_len:] if len(current_ids) > max_seq_len else current_ids
            x = torch.LongTensor([input_ids])
            s1_len_tensor = torch.LongTensor([min(s1_len, len(input_ids))])

            if torch.cuda.is_available():
                x = x.cuda()
                s1_len_tensor = s1_len_tensor.cuda()

            # 获取预测概率
            y_pred = model(x, s1_len_tensor)[0][-1]

            # 应用温度调整
            if temperature > 0:
                y_pred = y_pred / temperature
                y_pred = torch.softmax(y_pred, dim=-1)

            # 采样
            index = sampling_strategy(y_pred)

            # 检查是否结束
            if index == vocab["<EOS>"]:
                break

            current_ids.append(index)

            # 限制内容长度
            if len(current_ids) - s1_len > 300:
                break

        # 提取生成的内容部分
        content_ids = current_ids[s1_len:]
        generated_content = ''.join([reverse_vocab[id] for id in content_ids
                                     if id not in [vocab["<pad>"], vocab["<EOS>"]]])
        return generated_content


def sampling_strategy(prob_distribution):
    # 混合采样策略
    if random.random() < 0.8:  # 80%概率使用贪心
        return int(torch.argmax(prob_distribution))
    else:  # 20%概率使用随机采样
        prob = prob_distribution.cpu().numpy()
        prob = prob / prob.sum()  # 重新归一化
        return np.random.choice(len(prob), p=prob)


# 计算困惑度
def calc_perplexity(s1, s2, model, vocab, max_seq_len):
    prob = 0
    count = 0
    model.eval()
    with torch.no_grad():
        s1_ids = [vocab.get(c, vocab["<UNK>"]) for c in s1] + [vocab["<SEP>"]]
        s2_ids = [vocab.get(c, vocab["<UNK>"]) for c in s2] + [vocab["<EOS>"]]
        full_ids = s1_ids + s2_ids
        s1_len = len(s1_ids)

        for i in range(1, len(full_ids)):
            window = full_ids[:i]
            if len(window) > max_seq_len:
                window = window[-max_seq_len:]

            x = torch.LongTensor([window])
            s1_len_tensor = torch.LongTensor([min(s1_len, len(window))])
            target_index = full_ids[i]

            if torch.cuda.is_available():
                x = x.cuda()
                s1_len_tensor = s1_len_tensor.cuda()

            pred_prob = model(x, s1_len_tensor)[0][-1][target_index]
            prob += math.log(max(pred_prob.item(), 1e-10), 10)
            count += 1

    return 2 ** (-prob / count) if count > 0 else float('inf')


# 验证模型在验证集上的表现
def validate(model, val_x, val_y, val_s1_len, batch_size=32):
    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for i in range(0, len(val_x), batch_size):
            batch_x = val_x[i:i + batch_size]
            batch_y = val_y[i:i + batch_size]
            batch_s1 = val_s1_len[i:i + batch_size]

            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_s1 = batch_s1.cuda()

            loss = model(batch_x, batch_s1, batch_y)
            total_loss += loss.item() * len(batch_x)
            total_count += len(batch_x)

    avg_loss = total_loss / total_count
    avg_ppl = 2 ** avg_loss  # 从交叉熵损失计算困惑度
    return avg_loss, avg_ppl


def train(corpus_path, save_weight=True):
    # 超参数
    epoch_num = 20
    batch_size = 32
    train_sample = 30000  # 每轮训练样本数
    max_seq_len = 256
    val_ratio = 0.1  # 验证集比例
    best_val_loss = float('inf')

    # 加载数据
    vocab = build_vocab("vocab.txt")
    corpus = load_json_corpus(corpus_path)
    print(f"总语料数量: {len(corpus)}")

    if not corpus:
        print("错误：未加载到有效语料")
        return

    # 分割训练集和验证集
    train_corpus, val_corpus = split_train_val(corpus, val_ratio)
    print(f"训练集: {len(train_corpus)} 条, 验证集: {len(val_corpus)} 条")

    # 准备验证集数据
    val_x, val_y, val_s1_len = build_validation_dataset(vocab, max_seq_len, val_corpus)

    # 构建模型
    model = SFTLanguageModel(vocab)
    if torch.cuda.is_available():
        model = model.cuda()
        val_x, val_y, val_s1_len = val_x.cuda(), val_y.cuda(), val_s1_len.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    print("模型初始化完成，开始训练...")

    for epoch in range(epoch_num):
        # 训练阶段
        model.train()
        train_losses = []
        batches_per_epoch = int(train_sample / batch_size)

        for batch in range(batches_per_epoch):
            x, y, s1_lengths = build_sft_dataset(batch_size, vocab, max_seq_len, train_corpus)
            if torch.cuda.is_available():
                x, y, s1_lengths = x.cuda(), y.cuda(), s1_lengths.cuda()

            optim.zero_grad()
            loss = model(x, s1_lengths, y)
            loss.backward()
            optim.step()

            train_losses.append(loss.item())

            # 打印进度
            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}/{epoch_num}, Batch {batch}/{batches_per_epoch}, Loss: {loss.item():.4f}")

        # 验证阶段
        val_loss, val_ppl = validate(model, val_x, val_y, val_s1_len, batch_size)
        avg_train_loss = np.mean(train_losses)

        print(f"\nEpoch {epoch + 1} 总结:")
        print(f"训练集平均损失: {avg_train_loss:.4f}")
        print(f"验证集损失: {val_loss:.4f}, 验证集困惑度: {val_ppl:.2f}")

        # 测试生成效果
        test_titles = [s1 for s1, _ in random.sample(val_corpus, min(2, len(val_corpus)))]
        for title in test_titles:
            generated = generate_content_by_title(title, model, vocab, max_seq_len)
            print(f"\n测试标题: {title}")
            print(f"生成内容: {generated}")

        # 保存最优模型
        if val_loss < best_val_loss and save_weight:
            best_val_loss = val_loss
            os.makedirs("model", exist_ok=True)
            model_path = os.path.join("model", "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"最优模型已保存至 {model_path}")

    return model, vocab


# 独立的测试接口：加载模型并根据标题生成内容
def test_generate(title, model_path="model/best_model.pth", vocab_path="vocab.txt", max_seq_len=256):
    # 加载词表和模型
    vocab = build_vocab(vocab_path)
    model = SFTLanguageModel(vocab)

    # 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"警告：未找到模型文件 {model_path}，使用随机初始化模型")

    if torch.cuda.is_available():
        model = model.cuda()

    # 生成内容
    return generate_content_by_title(title, model, vocab, max_seq_len)


if __name__ == "__main__":
    # 训练模型
    model, vocab = train("sample_data.json", save_weight=True)

    # 示例：测试生成
    test_title = "阿根廷歹徒抢服装尺码不对拿回店里换"
    generated_content = test_generate(test_title)
    print(f"\n最终测试生成:")
    print(f"标题: {test_title}")
    print(f"生成内容: {generated_content}")
