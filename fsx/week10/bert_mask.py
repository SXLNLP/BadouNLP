import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os
import random
import numpy as np
from tqdm import tqdm
import math

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(
    '/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese')
bertModel = BertModel.from_pretrained('/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese',
                                      return_dict=False)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        # 加载预训练的BERT模型和tokenizer
        self.bert = bertModel
        self.tokenizer = tokenizer
        # 获取BERT的隐藏层维度
        hidden_size = self.bert.config.hidden_size
        # 分类器，用于预测masked token
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

        # 冻结BERT的参数，只训练分类器
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x, y=None):
        # x是token ids列表
        # 使用BERT获取上下文表示
        outputs = self.bert(input_ids=x)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # 应用dropout和分类器
        sequence_output = self.dropout(sequence_output)
        y_pred = self.classify(sequence_output)  # [batch_size, seq_len, vocab_size]

        if y is not None:
            # 计算masked language model的损失
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 返回预测的概率分布
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {}  # 预定义特殊token
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            # 从3开始分配索引，保留前3个位置给特殊token
            vocab[char] = index
    return vocab


# 加载语料 - 修改为使用UTF-8编码
def load_corpus(path):
    corpus = ""
    try:
        # 尝试使用UTF-8编码
        with open(path, encoding="utf-8") as f:
            for line in f:
                corpus += line.strip()
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试GBK
        with open(path, encoding="gbk") as f:
            for line in f:
                corpus += line.strip()
    return corpus


# 应用BERT风格的mask策略
def apply_bert_mask(tokens, vocab, mask_prob=0.15):
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)  # -100表示忽略该位置

    for i in range(1, len(tokens) - 1):  # 跳过[CLS]和[SEP]
        if random.random() < mask_prob:
            prob = random.random()
            labels[i] = tokens[i]  # 记录原始token作为标签

            if prob < 0.8:
                # 80%的概率替换为[MASK]
                masked_tokens[i] = vocab["[MASK]"]
            elif prob < 0.9:
                # 10%的概率替换为随机token
                masked_tokens[i] = random.randint(3, len(vocab) - 1)
            # 10%的概率保持不变

    return masked_tokens, labels


# 随机生成一个样本
def build_sample(vocab, window_size, corpus, tokenizer):
    start = random.randint(0, len(corpus) - window_size)
    end = start + window_size
    window = corpus[start:end]

    # 使用BERT tokenizer编码
    encoded = tokenizer.encode(window, add_special_tokens=True)

    # 应用mask策略
    masked_tokens, labels = apply_bert_mask(encoded, vocab)

    return masked_tokens, labels


# 建立数据集 - 优化为生成器方式
def build_dataset_generator(sample_length, vocab, window_size, corpus, tokenizer):
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus, tokenizer)
        yield x, y


# 建立数据集 - 批量处理版本
def build_batch_dataset(batch_size, vocab, window_size, corpus, tokenizer):
    dataset_x = []
    dataset_y = []

    # 生成一个批次的样本
    for x, y in build_dataset_generator(batch_size, vocab, window_size, corpus, tokenizer):
        dataset_x.append(x)
        dataset_y.append(y)

    # 填充到相同长度
    max_len = max(len(seq) for seq in dataset_x)
    padded_x = []
    padded_y = []

    for x, y in zip(dataset_x, dataset_y):
        pad_len = max_len - len(x)
        padded_x.append(x + [vocab["[PAD]"]] * pad_len)
        padded_y.append(y + [-100] * pad_len)  # -100表示忽略填充位置

    return torch.LongTensor(padded_x), torch.LongTensor(padded_y)


# 建立模型
def build_model(vocab):
    model = LanguageModel(len(vocab))
    return model


# 文本生成测试代码 - 改进版本
def generate_sentence(prefix, model, max_length=30):
    tokenizer = model.tokenizer
    model.eval()

    # 编码前缀
    input_ids = tokenizer.encode(prefix, add_special_tokens=True)
    input_ids = torch.LongTensor([input_ids])

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        # 生成文本直到达到最大长度或生成结束标记
        for _ in range(max_length):
            # 获取模型预测
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]

            # 采样下一个token - 使用更高级的采样策略
            # 这里使用top-k采样
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            top_k_probs = torch.softmax(top_k_logits, dim=-1)

            # 从top-k中随机采样
            next_token_id = top_k_indices[torch.multinomial(top_k_probs, 1)].item()

            # 如果生成了结束标记，就停止
            if next_token_id == tokenizer.sep_token_id:
                break

            # 添加到输入序列
            input_ids = torch.cat([input_ids, torch.LongTensor([[next_token_id]])], dim=1)

        # 转换为文本
        generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

    return generated_text


# 计算文本ppl
def calc_perplexity(sentence, model):
    tokenizer = model.tokenizer
    model.eval()

    # 编码句子
    encoded = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.LongTensor([encoded])

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        # 获取模型预测
        outputs = model(input_ids)
        logits = outputs[0]  # [seq_len, vocab_size]

        # 计算每个token的概率
        probs = torch.softmax(logits, dim=-1)

        # 计算perplexity
        log_probs = []
        for i in range(1, len(encoded)):
            token_id = encoded[i]
            # 获取前一个位置对当前token的预测概率
            prob = probs[i - 1, token_id].item()
            log_probs.append(math.log(prob))

        # 计算平均log概率并转换为perplexity
        if not log_probs:
            return float('inf')

        avg_log_prob = sum(log_probs) / len(log_probs)
        ppl = math.exp(-avg_log_prob)

        return ppl


def train(corpus_path, save_weight=True):
    epoch_num = 3  # 训练轮数
    batch_size = 16  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    window_size = 50  # 样本文本长度

    # 创建保存模型的目录
    os.makedirs("model", exist_ok=True)

    # 建立字表
    vocab = build_vocab("/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese/vocab.txt")
    print(f"词汇表大小: {len(vocab)}")

    # 加载语料
    corpus = load_corpus(corpus_path)
    print(f"语料长度: {len(corpus)}")

    # 建立模型
    model = build_model(vocab)
    if torch.cuda.is_available():
        model = model.cuda()

    # 建立优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 使用tqdm显示进度条
        progress_bar = tqdm(range(int(train_sample / batch_size)), desc=f"Epoch {epoch + 1}/{epoch_num}")

        for _ in progress_bar:
            # 构建一组训练样本
            x, y = build_batch_dataset(batch_size, vocab, window_size, corpus, model.tokenizer)

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            watch_loss.append(loss.item())
            progress_bar.set_postfix({"loss": np.mean(watch_loss)})

        print(f"\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")

        # 生成一些文本示例
        prefixes = [
            "让他在半年之前，就不能做出",
            "李慕站在山路上，深深的呼吸"
        ]

        for prefix in prefixes:
            generated = generate_sentence(prefix, model)
            print(f"生成示例: {generated}")

        # 计算perplexity
        sample_text = corpus[:100]  # 取一小段语料作为样本
        ppl = calc_perplexity(sample_text, model)
        print(f"样本perplexity: {ppl:.4f}")

    if save_weight:
        base_name = os.path.basename(corpus_path).replace(".txt", ".pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")

    return model


if __name__ == "__main__":
    # 确保vocab.txt文件存在
    vocab_path = "/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese/vocab.txt"
    if not os.path.exists(vocab_path):
        print(f"错误: vocab.txt文件不存在! 请检查路径: {vocab_path}")
    else:
        # 检查语料文件
        corpus_path = "corpus.txt"
        if not os.path.exists(corpus_path):
            print(f"警告: 语料文件 {corpus_path} 不存在，请确保文件存在后再运行")
        else:
            train(corpus_path)
