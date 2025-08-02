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

"""
基于pytorch的BERT文本生成模型
功能：根据标题生成内容
"""

# 定义模型路径和数据路径，请根据你的实际情况修改
bert_path = r"D:\models\google-bert\bert-base-chinese"
data_path = r"D:\人工智能\week10 文本生成问题\transformers-生成文章标题\sample_data.json"


class LanguageModel(nn.Module):
    # 修改点 1: __init__ 方法的参数列表已完整，保留
    def __init__(self, hidden_size, vocab_size, pretrain_model_path, tokenizer, max_title_len, max_content_len):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)

        # 确保使用 nn.CrossEntropyLoss 实例，并设置 ignore_index=-100
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        self.tokenizer = tokenizer
        self.sep_idx = max_title_len
        self.max_content_len = max_content_len

    def forward(self, x, y=None):  # x 实际是 input_ids
        device = x.device
        batch_size, seq_len = x.shape

        pad_token_id = self.tokenizer.pad_token_id

        initial_padding_mask = (x != pad_token_id).long()

        # 注意力掩码的逻辑上次确认是正确的，不需要修改
        custom_attention_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=device))

        for i in range(batch_size):
            custom_attention_mask[i, 0:self.sep_idx + 1, 0:self.sep_idx + 1] = 1.0

        final_attention_mask = (custom_attention_mask *
                                initial_padding_mask.unsqueeze(1).float() *
                                initial_padding_mask.unsqueeze(2).float())

        x_bert_output, _ = self.bert(x, attention_mask=final_attention_mask)

        y_pred = self.classify(x_bert_output)

        if y is not None:
            return self.loss_fct(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


# 随机生成一个样本
# 输入格式: [title_tokens (padded)] + [SEP] + [content_tokens (padded)]
# Labels格式: [-100]*len(title_tokens) + [-100] + [content_tokens (PAD也为-100)]
def build_sample(tokenizer, max_content_len, max_title_len, samples):
    data = random.choice(samples)
    title, content = data["title"], data["content"],

    x_ids = tokenizer.encode(title, add_special_tokens=False, padding='max_length', truncation=True,
                             max_length=max_title_len)
    y_ids = tokenizer.encode(content, add_special_tokens=False, padding='max_length', truncation=True,
                             max_length=max_content_len)

    input_ids = x_ids + [tokenizer.sep_token_id] + y_ids

    labels = [-100] * len(x_ids)
    labels.append(-100)

    # 核心修复点 2: 确保内容部分中的 PAD token 在 labels 中也为 -100
    for token_id in y_ids:
        if token_id == tokenizer.pad_token_id:
            labels.append(-100)
        else:
            labels.append(token_id)

    return input_ids, labels


# 建立数据集
def build_dataset(sample_length, tokenizer, max_content_len, max_title_len, samples):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, max_content_len, max_title_len, samples)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 文本生成测试代码
def generate_content(input_title, model, tokenizer, max_title_len, max_generate_len=50, debug=False):
    model.eval()

    with torch.no_grad():
        device = next(model.parameters()).device

        input_title_ids = tokenizer.encode(
            input_title,
            add_special_tokens=False,
            truncation=True,
            padding='max_length',
            max_length=max_title_len
        )

        # 初始序列：[标题ID (padded)] + [SEP]
        current_input_sequence_ids = input_title_ids + [tokenizer.sep_token_id]

        generated_content_ids = []
        # 模型输入总长度 = max_title_len (标题) + 1 (SEP) + max_content_len (内容)
        total_model_input_len = max_title_len + 1 + model.max_content_len

        for i in range(max_generate_len):  # 迭代生成，最多生成 max_generate_len 个 token
            # 如果已生成内容的长度达到 max_content_len，停止生成
            if len(generated_content_ids) >= model.max_content_len:
                if debug: print(f"达到最大内容生成长度 {model.max_content_len}，停止。")
                break

            # 构建送入模型的输入序列：[标题ID] + [SEP] + [已生成内容ID]
            # 这个序列是实际的有效 token 部分
            effective_input_sequence = current_input_sequence_ids + generated_content_ids

            # 将 effective_input_sequence 填充到 total_model_input_len
            # 这确保了每次模型接收的 input_ids 形状都一致
            padded_input_ids = effective_input_sequence[:]  # 复制一份，避免修改原列表
            padded_input_ids.extend([tokenizer.pad_token_id] * (total_model_input_len - len(padded_input_ids)))

            input_tensor = torch.LongTensor([padded_input_ids]).to(device)

            # 核心修复点 3: 获取正确位置的 logits
            # 我们需要预测的是 `effective_input_sequence` 中最后一个 token 的下一个 token。
            # BERT 模型的输出 `x_bert_output[0, k]` 是对 `input_ids[k]` 这个 token 的表示。
            # 要预测下一个 token，我们需要使用 `effective_input_sequence` 最后一个 token (索引为 `len(effective_input_sequence) - 1`) 的表示。
            # 所以，我们要取 `x_bert_output[0, len(effective_input_sequence) - 1]` 对应的分类层输出。

            # 例：[T, T, SEP, C_0, C_1] -> 长度 L
            # 我们需要 C_1 的输出 logits 来预测 C_2
            # C_1 的索引是 L-1

            pred_logits = model.classify(model.bert(input_tensor, attention_mask=None)[0])  # 重新调用BERT获取output
            # 也可以直接调用model(input_tensor)来获取softmax前的logits，但这里为了调试方便单独拆开

            # 确保获取的是有效输入序列的最后一个 token 的 logits
            logits_for_next_token = pred_logits[0, len(effective_input_sequence) - 1, :]

            pred_prob = torch.softmax(logits_for_next_token, dim=-1)  # 应用softmax得到概率分布

            if debug and i < 5:  # 仅在调试模式下打印前几个生成步骤
                top5_probs, top5_indices = torch.topk(pred_prob, 5)
                decoded_top5 = [tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in top5_indices]
                print(f"--- 生成步 {i + 1} ---")
                print(f"当前输入有效长度: {len(effective_input_sequence)}")
                print(f"预测位置索引: {len(effective_input_sequence) - 1}")
                print(f"最高概率: {top5_probs[0].item():.4f}, 预测token: {decoded_top5[0]}")
                print(f"Top 5 预测: {list(zip(decoded_top5, top5_probs.tolist()))}")

            next_token_id = sampling_strategy(pred_prob)

            # 遇到结束符（SEP或PAD）则停止生成
            if next_token_id == tokenizer.sep_token_id:  # 停止生成内容，如果是SEP token
                if debug: print(f"生成SEP token {next_token_id}，停止生成。")
                break
            if next_token_id == tokenizer.pad_token_id:  # 停止生成内容，如果是PAD token
                if debug: print(f"生成PAD token {next_token_id}，停止生成。")
                break

            generated_content_ids.append(next_token_id)

        # 返回生成的文本
        return tokenizer.decode(generated_content_ids, skip_special_tokens=True).replace(" ", "")

    # 采样策略（平衡随机性和准确性）


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:  # 90%概率贪婪采样（选概率最高的token）
        return int(torch.argmax(prob_distribution))
    else:  # 10%概率随机采样（增加多样性）
        prob = prob_distribution.cpu().numpy().astype(np.float64)
        # 修复点 4: 更稳健的归一化处理，确保概率和为1
        prob = prob / (prob.sum() + 1e-9)  # 加一个小的epsilon防止除以0
        # 如果归一化后仍有问题，或者prob全0，则平均分配
        if np.isclose(prob.sum(), 0):
            prob = np.ones_like(prob) / len(prob)
        return np.random.choice(len(prob), p=prob)


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # BERT-base 的 hidden_size
    vocab_size = 21128  # BERT-base-chinese 的字表大小

    # 核心修复点 5: 降低学习率，这是解决模型收敛到“镑”等重复token的关键一步
    learning_rate = 2e-5  # 推荐将学习率降低到 1e-5 或 2e-5 进行微调

    # 明确 max_title_len 和 max_content_len，确保与 build_sample 一致
    max_title_len = 20  # 标题最大长度（包含padding）
    max_content_len = 100  # 内容最大长度（包含padding）

    pretrain_model_path = bert_path
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)

    # 确保 LanguageModel 实例化时传入 max_title_len 和 max_content_len
    model = LanguageModel(char_dim, vocab_size, pretrain_model_path, tokenizer, max_title_len, max_content_len)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("文本生成模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, max_content_len, max_title_len, corpus)
            x, y = x.to(device), y.to(device)

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 修改点 6: 在生成时开启调试模式，观察前几步的预测详情
        print("生成内容1:",
              generate_content("让他在半年之前，就不能做出", model, tokenizer, max_title_len, max_generate_len=50,
                               debug=True if epoch == 0 else False))
        print("生成内容2:",
              generate_content("李慕站在山路上，深深的呼吸", model, tokenizer, max_title_len, max_generate_len=50,
                               debug=False))

    if not save_weight:
        return
    else:
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        base_name = os.path.basename(corpus_path).replace(".json", ".pth")
        model_path = os.path.join(model_dir, base_name)
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存到: {model_path}")
        return


if __name__ == "__main__":
    train(data_path, True)

