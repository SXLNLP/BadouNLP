# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
# 关闭SSL验证（如果需要）
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from transformers import BertModel, BertTokenizer




class BertLanguageModel(nn.Module):
    def __init__(self, tokenizer, model_path):
        super(BertLanguageModel, self).__init__()
        # 加载本地BERT模型
        self.bert = BertModel.from_pretrained(model_path)
        # 分类头：使用BERT默认的hidden_size
        self.classifier = nn.Linear(self.bert.config.hidden_size, tokenizer.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充标签

    def forward(self, input_ids, attention_mask, labels=None):
        # 兼容旧版本transformers：BertModel返回元组，第一个元素是last_hidden_state
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 旧版本返回格式：(last_hidden_state, pooler_output, ...)
        sequence_output = bert_outputs[0]  # 用索引获取，替代属性访问

        # 预测所有位置的token
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, vocab_size)

        if labels is not None:
            # 展平计算损失（忽略-100的标签）
            return self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            return torch.softmax(logits, dim=-1)


# 加载语料（自动适配编码）
def load_corpus(path):
    corpus = ""
    try:
        with open(path, encoding="utf8") as f:
            for line in f:
                corpus += line.strip() + "\n"
    except UnicodeDecodeError:
        with open(path, encoding="gbk") as f:
            for line in f:
                corpus += line.strip() + "\n"
    return corpus


# 构建MLM样本（增加attention_mask生成）
def build_sample(tokenizer, max_seq_len, corpus):
    # 随机截取一段文本
    if len(corpus) <= max_seq_len:
        start = 0
    else:
        start = random.randint(0, len(corpus) - max_seq_len)
    text = corpus[start:start + max_seq_len]

    # 分词
    tokens = tokenizer.tokenize(text)
    if not tokens:
        tokens = [tokenizer.unk_token]

    # 截断或填充到指定长度
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    else:
        tokens = tokens + [tokenizer.pad_token] * (max_seq_len - len(tokens))

    # 转换为id
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    labels = input_ids.copy()

    # 生成attention_mask（1表示有效token，0表示填充）
    attention_mask = [1 if token != tokenizer.pad_token else 0 for token in tokens]

    # 执行MLM掩码
    for i in range(len(input_ids)):
        if input_ids[i] == tokenizer.pad_token_id:
            labels[i] = -100  # 填充位置不计算损失
            continue

        if random.random() < 0.15:  # 15%概率选中
            # 80%概率替换为[MASK]
            if random.random() < 0.8:
                input_ids[i] = tokenizer.mask_token_id
            # 10%概率替换为随机token
            elif random.random() < 0.5:
                input_ids[i] = random.randint(5, tokenizer.vocab_size - 1)  # 避开特殊token
            # 10%概率保持不变
        else:
            labels[i] = -100  # 未选中的位置不计算损失

    # 确保至少有一个有效标签
    if all(l == -100 for l in labels):
        pos = random.randint(0, len(labels) - 1)
        labels[pos] = input_ids[pos]
        input_ids[pos] = tokenizer.mask_token_id

    return input_ids, attention_mask, labels


# 构建数据集（增加attention_mask）
def build_dataset(sample_num, tokenizer, max_seq_len, corpus):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for _ in range(sample_num):
        input_ids, attention_mask, labels = build_sample(tokenizer, max_seq_len, corpus)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return (
        torch.LongTensor(input_ids_list),
        torch.LongTensor(attention_mask_list),
        torch.LongTensor(labels_list)
    )


# 文本生成函数
def generate_sentence(openings, model, tokenizer, max_seq_len, max_gen_len=30):
    model.eval()
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(openings))

    with torch.no_grad():
        for _ in range(max_gen_len):
            # 截断到最大长度
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[-max_seq_len:]

            # 生成attention_mask
            attention_mask = [1] * len(input_ids)
            # 填充到最大长度（如果需要）
            if len(input_ids) < max_seq_len:
                pad_len = max_seq_len - len(input_ids)
                input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
                attention_mask = [0] * pad_len + attention_mask

            # 转换为tensor
            input_tensor = torch.LongTensor([input_ids])
            attention_tensor = torch.LongTensor([attention_mask])

            # 获取预测
            logits = model(input_tensor, attention_tensor)[0, -1, :]  # 取最后一个token的预测

            # 采样下一个token
            if random.random() > 0.1:
                next_token_id = torch.argmax(logits).item()
            else:
                probs = torch.softmax(logits, dim=-1).numpy()
                next_token_id = np.random.choice(len(probs), p=probs)

            input_ids.append(next_token_id)

            # 结束条件
            next_token = tokenizer.decode([next_token_id])
            if next_token in ["。", "！", "？"] and len(input_ids) > len(openings) + 5:
                break

    return tokenizer.decode(input_ids).replace(" ", "")  # 去除空格


# 训练函数（适配CPU，增加attention_mask处理）
def train(corpus_path, save_weight=False):
    # 本地BERT模型路径（修改为你的实际路径）
    local_model_path = r"D:\BaiduNetdiskDownload\组件\ppt\AI\数据处理与统计分析\bert-base-chinese"  # 替换为你的模型文件夹路径

    # 超参数（CPU训练建议减小batch_size）
    epoch_num = 3  # CPU训练较慢，减少轮次
    batch_size = 8  # CPU训练减小batch_size
    train_sample = 4000  # 减少每轮样本数
    max_seq_len = 32  # 减小序列长度，降低CPU负担
    lr = 2e-5

    # 加载本地分词器
    tokenizer = BertTokenizer.from_pretrained(local_model_path)

    # 加载语料
    corpus = load_corpus(corpus_path)
    if not corpus:
        raise ValueError("请检查corpus.txt是否存在且内容不为空")
    print(f"成功加载语料，长度：{len(corpus)}")

    # 构建模型（CPU模式）
    model = BertLanguageModel(tokenizer, local_model_path)
    device = "cpu"  # 强制使用CPU
    model.to(device)
    print(f"使用{device}训练")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("模型加载完成，开始训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        batch_count = 0

        # 每轮训练
        for _ in range(train_sample // batch_size):
            # 获取批次数据（包含attention_mask）
            input_ids, attention_mask, labels = build_dataset(
                batch_size, tokenizer, max_seq_len, corpus)

            # 转移到设备（CPU）
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 训练步骤
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)  # 传入attention_mask
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # 打印进度
            if (batch_count + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_count + 1}, Loss: {loss.item():.4f}")

        # 每轮结束打印信息
        avg_loss = total_loss / batch_count
        print(f"===== 第{epoch + 1}轮平均Loss: {avg_loss:.4f} =====")

        # 生成示例
        print("生成示例1:", generate_sentence("让他在半年之前，就不能做出", model, tokenizer, max_seq_len))
        print("生成示例2:", generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, max_seq_len))

    if save_weight:
        if not os.path.exists("model"):
            os.makedirs("model")
        torch.save(model.state_dict(), "model/bert_mlm_cpu.pth")
        print("模型已保存至 model/bert_mlm_cpu.pth")


if __name__ == "__main__":
    train("corpus.txt", False)
