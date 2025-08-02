# coding:utf8
import torch
import torch.nn as nn
import random
import os
import json
from transformers import BertTokenizer, BertModel


class SFTModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(SFTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if y is not None:
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


# 加载训练数据
def load_sft_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            title = item["title"]
            content = item["content"]
            # 将标题和内容组合成训练样本
            data.append(f"{title}[SEP]{content}")
    return data


# 构建SFT训练样本
def build_sft_sample(tokenizer, max_length, corpus):
    # 随机选择一个样本
    text = random.choice(corpus)

    # 使用tokenizer处理文本
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].squeeze(0)

    # 创建标签 - 右移一位
    labels = input_ids.clone()
    labels[:-1] = input_ids[1:].clone()
    labels[-1] = tokenizer.pad_token_id  # 最后一个位置用pad填充

    return input_ids, labels


# 构建SFT数据集
def build_sft_dataset(sample_length, tokenizer, max_length, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sft_sample(tokenizer, max_length, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.stack(dataset_y)


# 文本生成测试（适配SFT）
def generate_sft_text(prompt, model, tokenizer, max_length):
    model.eval()
    with torch.no_grad():
        generated = prompt
        input_ids = tokenizer.encode(generated, return_tensors='pt', add_special_tokens=True)

        # 生成文本
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0)

            # 如果生成了[SEP]或[PAD]，停止生成
            if next_token.item() in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                break
            generated += tokenizer.decode(next_token.cpu().numpy())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # 提取生成的内容部分
        sep_pos = generated.find("[SEP]")
        if sep_pos != -1:
            generated = generated[sep_pos + 5:]  # 移除标题部分

        return generated


def train_sft(corpus_path, pretrain_model_path, save_weight=True):
    epoch_num = 10  # 训练轮数
    batch_size = 32  # 批量大小
    train_sample = 10000  # 每轮训练样本数
    max_length = 64  # 最大序列长度
    learning_rate = 1e-5  # 学习率
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    # 添加特殊token（如果不存在）
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    # 加载训练数据
    corpus = load_sft_data(corpus_path)
    # 创建模型
    vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)
    model = SFTModel(768, vocab_size, pretrain_model_path)
    # 优化器
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("开始SFT训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch in range(int(train_sample / batch_size)):
            # 构建训练批次
            x, y = build_sft_dataset(batch_size, tokenizer, max_length, corpus)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

            if batch % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch}/{int(train_sample / batch_size)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / (train_sample / batch_size)
        print(f"=========\nEpoch {epoch + 1} 平均损失: {avg_loss:.4f}")

        # 测试生成
        test_prompt = "美国最适合创业的十大行业"
        generated = generate_sft_text(test_prompt, model, tokenizer, 50)
        print(f"生成示例:\n输入: {test_prompt}\n输出: {generated}")

    if save_weight:
        model_path = os.path.join("sft_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")


# 配置路径
data_path = r"C:\BaiduNetdiskDownload\八斗精品班\week10 文本生成问题\transformers-生成文章标题\sample_data.json"  # 替换为你的JSON数据路径
bert_path = r"C:\BaiduNetdiskDownload\八斗精品班\week6 语言模型和预训练\bert-base-chinese";
train_sft(data_path, bert_path)
