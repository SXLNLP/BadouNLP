# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from transformers import BertTokenizer, BertModel

"""
基于SFT（监督微调）的文章标题生成模型
功能：输入文章content，生成对应的title
"""


class TitleGenerationModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path, tokenizer):
        super(TitleGenerationModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)  # 从BERT隐藏层映射到词表
        self.loss = nn.functional.cross_entropy
        self.tokenizer = tokenizer  # 保存tokenizer实例（解决NameError）

    def forward(self, input_ids, labels=None):
        # 输入格式：[content_ids] + [SEP] + [title_ids]（训练）；[content_ids] + [SEP]（预测）
        if labels is not None:
            # 1. 构建SFT掩码（标题只能看到content和前缀）
            batch_size, seq_len = input_ids.shape
            mask = torch.tril(torch.ones((batch_size, seq_len, seq_len), device=input_ids.device))

            # 2. BERT编码（带掩码）
            output, _ = self.bert(input_ids, attention_mask=mask)

            # 3. 计算标题部分损失（忽略content和SEP）
            sep_token_id = self.tokenizer.sep_token_id  # 使用实例化的tokenizer
            sep_positions = (input_ids == sep_token_id).nonzero()[:, 1]  # 每个样本的SEP位置
            total_loss = 0.0

            for i in range(batch_size):
                sep_pos = sep_positions[i]
                # 截取标题部分（SEP之后）
                title_pred = self.classify(output[i, sep_pos + 1:])  # 预测值
                title_true = labels[i, sep_pos + 1:]  # 真实标签
                # 过滤padding（标签为-100的位置不参与损失计算）
                mask_title = (title_true != -100)
                if mask_title.any():  # 避免空标题导致的错误
                    total_loss += self.loss(
                        title_pred[mask_title],
                        title_true[mask_title]
                    )

            return total_loss / batch_size  # 平均损失
        else:
            # 预测时：输入[content] + [SEP]，生成标题
            output, _ = self.bert(input_ids)
            pred = self.classify(output)
            return torch.softmax(pred, dim=-1)


# 加载JSON Lines格式数据（每行一个JSON对象）
def load_json_data(json_path):
    """适配多行独立JSON对象的格式，每行一个{"title":..., "content":...}"""
    samples = []
    with open(json_path, encoding="utf8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                item = json.loads(line)  # 解析单行JSON
                # 验证字段完整性
                if "title" in item and "content" in item and item["title"] and item["content"]:
                    samples.append({
                        "content": item["content"].strip(),
                        "title": item["title"].strip()
                    })
                else:
                    print(f"警告：第{line_num}行缺少title或content字段，已跳过")
            except json.JSONDecodeError as e:
                print(f"错误：第{line_num}行JSON格式错误 - {e}，已跳过")
    return samples


# 构建单个训练样本
def build_sample(tokenizer, max_content_len, max_title_len, data_samples):
    sample = random.choice(data_samples)
    content, title = sample["content"], sample["title"]

    # 1. 编码content和title
    content_ids = tokenizer.encode(
        content,
        add_special_tokens=False,
        truncation=True,
        max_length=max_content_len
    )
    title_ids = tokenizer.encode(
        title,
        add_special_tokens=False,
        truncation=True,
        max_length=max_title_len
    )

    # 2. 拼接输入：content + [SEP] + title
    input_ids = content_ids + [tokenizer.sep_token_id] + title_ids
    # 3. 构建标签：content和SEP用-100标记（不参与损失）
    labels = [-100] * len(content_ids) + [-100] + title_ids  # -100是PyTorch忽略损失的标记

    # 4. 补齐长度（统一到max_total_len）
    max_total_len = max_content_len + 1 + max_title_len  # +1是SEP
    if len(input_ids) < max_total_len:
        pad_len = max_total_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len  # padding不参与损失
    else:
        input_ids = input_ids[:max_total_len]
        labels = labels[:max_total_len]

    return input_ids, labels


# 构建数据集
def build_dataset(sample_num, tokenizer, max_content_len, max_title_len, data_samples):
    dataset_input = []
    dataset_label = []
    for _ in range(sample_num):
        input_ids, labels = build_sample(tokenizer, max_content_len, max_title_len, data_samples)
        dataset_input.append(input_ids)
        dataset_label.append(labels)
    return torch.LongTensor(dataset_input), torch.LongTensor(dataset_label)


# 标题生成函数（自回归生成）
def generate_title(content, model, tokenizer, max_content_len, max_title_len):
    model.eval()
    with torch.no_grad():
        # 1. 编码content并拼接SEP
        content_ids = tokenizer.encode(
            content,
            add_special_tokens=False,
            truncation=True,
            max_length=max_content_len
        )
        input_ids = content_ids + [tokenizer.sep_token_id]  # 输入格式：[content] + [SEP]
        input_tensor = torch.LongTensor([input_ids]).to(next(model.parameters()).device)

        # 2. 自回归生成标题（最多生成max_title_len个token）
        for _ in range(max_title_len):
            pred_prob = model(input_tensor)[0, -1]  # 取最后一个位置的预测
            next_token_id = sampling_strategy(pred_prob)

            # 遇到结束符（SEP或PAD）则停止
            if next_token_id in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                break

            # 拼接新token继续生成
            input_ids.append(next_token_id)
            input_tensor = torch.LongTensor([input_ids]).to(next(model.parameters()).device)

        # 3. 提取标题（SEP之后的内容）
        sep_pos = input_ids.index(tokenizer.sep_token_id)
        title_ids = input_ids[sep_pos + 1:]
        return tokenizer.decode(title_ids).replace(" ", "")  # 去除多余空格


# 采样策略（平衡随机性和准确性）
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:  # 90%概率贪婪采样（选概率最高的token）
        return int(torch.argmax(prob_distribution))
    else:  # 10%概率随机采样（增加多样性）
        prob = prob_distribution.cpu().numpy()
        return np.random.choice(len(prob), p=prob)


# 训练主函数
def train(json_path, save_weight=True):
    # 超参数
    epoch_num = 30
    batch_size = 16
    train_sample_num = 1000  # 每轮训练样本数（根据数据量调整）
    max_content_len = 200  # 文章内容最大长度
    max_title_len = 30  # 标题最大长度
    hidden_size = 768  # BERT隐藏层维度
    learning_rate = 2e-5  # SFT推荐较小的学习率

    # 加载预训练模型和分词器
    pretrain_model_path = r'D:\BaiduNetdiskDownload\AI架构课程\第六周 语言模型\week6\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    # 设置PAD token（BERT默认无PAD，用EOS代替）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    data_samples = load_json_data(json_path)
    if not data_samples:
        raise ValueError("未加载到有效样本，请检查JSON文件")
    print(f"成功加载{len(data_samples)}个样本，开始SFT训练...")

    # 初始化模型（传入tokenizer解决NameError）
    model = TitleGenerationModel(
        hidden_size=hidden_size,
        vocab_size=tokenizer.vocab_size,
        pretrain_model_path=pretrain_model_path,
        tokenizer=tokenizer  # 关键：将tokenizer传入模型
    )
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU加速训练")
    else:
        print("使用CPU训练")

    # 优化器（SFT推荐AdamW）
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        steps_per_epoch = train_sample_num // batch_size

        for batch_idx in range(steps_per_epoch):
            # 生成batch数据
            input_ids, labels = build_dataset(
                batch_size,
                tokenizer,
                max_content_len,
                max_title_len,
                data_samples
            )
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()

            # 计算损失并更新参数
            optimizer.zero_grad()
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 打印中间进度
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epoch_num}, Batch {batch_idx + 1}/{steps_per_epoch}, Loss: {loss.item():.4f}")

        # 打印本轮平均损失
        avg_loss = total_loss / steps_per_epoch
        print(f"\n========= 第{epoch + 1}轮训练结束 =========")
        print(f"平均Loss: {avg_loss:.4f}")

        # 测试生成效果（用第一个样本的content）
        test_content = data_samples[0]["content"]
        generated_title = generate_title(
            test_content,
            model,
            tokenizer,
            max_content_len,
            max_title_len
        )
        print(f"测试生成标题：{generated_title}")
        print(f"真实标题：{data_samples[0]['title']}\n")

    # 保存模型
    if save_weight:
        os.makedirs("title_model", exist_ok=True)
        model_path = os.path.join("title_model", "sft_title_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至：{model_path}")
    return model


if __name__ == "__main__":
    # 替换为你的JSON文件路径（确保是每行一个JSON对象的格式）
    json_path = r"D:\BaiduNetdiskDownload\AI架构课程\第十一周 大模型相关内容第一讲\sample_data.json"  # 例如：包含你提供的4条新闻数据的文件
    train(json_path, save_weight=True)
