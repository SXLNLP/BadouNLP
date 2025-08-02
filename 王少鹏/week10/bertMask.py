# coding:utf-8
import os
import math
import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertLMHeadModel

# 模型与训练相关配置
MODEL_NAME = "bert-base-chinese"          # 使用中文预训练BERT模型名称
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测是否使用GPU
BATCH_SIZE = 16                            # 每个batch中样本数量
EPOCHS = 3                                 # 训练轮数
LR = 5e-5                                  # 学习率
WINDOW_SIZE = 12                           # 窗口大小：自回归预测的上下文长度
TRAIN_SAMPLES = 1000                       # 用于训练的总样本数

# 自定义数据集类，用于样本生成与batch管理
class BERTTextDataset(Dataset):
    def __init__(self, tokenizer, window_size, corpus, sample_length):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.corpus = corpus
        self.sample_length = sample_length
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return self.sample_length

    def __getitem__(self, idx):
        """
        生成一个样本对：输入窗口 和 对应的预测目标窗口（错位）
        """
        start = random.randint(0, len(self.corpus) - self.window_size - 2)
        window = self.corpus[start: start + self.window_size]
        target = self.corpus[start + 1: start + self.window_size + 1]

        x = self.tokenizer.encode(window, add_special_tokens=False)
        y = self.tokenizer.encode(target, add_special_tokens=False)

        # padding到window_size长度
        x += [self.pad_id] * (self.window_size - len(x))
        y += [self.pad_id] * (self.window_size - len(y))

        return torch.tensor(x), torch.tensor(y)


# 自回归语言模型BERT（微调结构）
class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 加载BERT配置并把其debug为decoder风格
        config = BertConfig.from_pretrained(MODEL_NAME)
        config.is_decoder = True
        config.add_cross_attention = False
        config.vocab_size = vocab_size
        self.model = BertLMHeadModel(config)

    def forward(self, x, y=None):
        # 获取BERT输出的logits
        outputs = self.model(input_ids=x)
        logits = outputs.logits  # shape: (B, T, V)
        if y is not None:
            # 若有目标标签则计算交叉熵损失
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            return loss
        else:
            # 推理时输出Softmax概率分布
            return torch.softmax(logits, dim=-1)


# 加载训练语料文本
def load_corpus(path, encoding="utf-8"):
    with open(path, encoding=encoding) as f:
        return f.read().replace("\n", "")


# 基于当前模型和起始句子生成文本
def generate_sentence(opening, model, tokenizer, window_size, max_len=50):
    model.eval()
    text = opening
    with torch.no_grad():
        while not text.endswith("。") and len(text) < max_len:
            # 获取窗口范围内的输入
            x = tokenizer.encode(text[-window_size:], add_special_tokens=False)
            x += [tokenizer.pad_token_id] * (window_size - len(x))
            x_tensor = torch.tensor([x], device=DEVICE)
            # 获取最后一个位置的输出分布
            probs = model(x_tensor)[0, -1]
            # 采用采样策略选出下一个token
            next_id = sample_from_probs(probs)
            next_token = tokenizer.decode([next_id]).replace(" ", "")
            text += next_token
    return text


# 采样策略：90%贪婪，10%采样
def sample_from_probs(prob_dist):
    if random.random() > 0.1:
        return int(torch.argmax(prob_dist))  # 贪婪选最大概率
    else:
        dist = prob_dist.clone()
        dist = dist / dist.sum()  # 归一化为概率分布
        return int(torch.multinomial(dist, 1))  # 按概率采样


# 基于句子计算Perplexity评估指标
def calc_perplexity(sentence, model, tokenizer, window_size=12):
    model.eval()
    encoded = tokenizer.encode(sentence, add_special_tokens=False)
    log_prob_sum = 0
    with torch.no_grad():
        for i in range(1, len(encoded)):
            start = max(0, i - window_size)
            x = encoded[start:i]
            x += [tokenizer.pad_token_id] * (window_size - len(x))
            x_tensor = torch.tensor([x], device=DEVICE)
            probs = model(x_tensor)[0, -1]
            prob = probs[encoded[i]].item()
            log_prob_sum += math.log(prob + 1e-10)  # 避免log(0)
    avg_log_prob = log_prob_sum / len(encoded)
    return math.exp(-avg_log_prob)


# 主训练函数
def train(corpus_path, save_model=True):
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # 加载文本语料
    corpus = load_corpus(corpus_path, encoding="gbk")  # 使用GBK编码文本

    # 准备数据集和dataloader
    dataset = BERTTextDataset(tokenizer, WINDOW_SIZE, corpus, TRAIN_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 构建模型和优化器
    model = LanguageModel(vocab_size=tokenizer.vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("模型初始化完成，开始训练")
    for epoch in range(EPOCHS):
        model.train()
        losses = []

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            loss = model(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"第 {epoch + 1}/{EPOCHS} 轮训练完成，平均损失：{np.mean(losses):.4f}")
        # 每轮结束生成示例句子
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, WINDOW_SIZE))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, WINDOW_SIZE))

    # 保存训练好的模型权重
    if save_model:
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "./model/bert_lm.pth")
        print("模型保存完成，路径：model/bert_lm.pth")


# 运行主程序
if __name__ == "__main__":
    train("corpus.txt")