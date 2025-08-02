# coding: utf8
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import os
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import softmax
"""
SFT 训练脚本：基于 BERT + causal attention 实现自回归语言模型
语料格式：news_theme_sft.json
"""


class CausalBertLM(nn.Module):
    def __init__(self, vocab_size, pretrain_model_path):
        super(CausalBertLM, self).__init__()
        # 修改 config，启用 is_decoder（允许 causal attention）
        config = BertConfig.from_pretrained(pretrain_model_path)
        config.is_decoder = True  # 关键：启用 decoder 模式
        config.add_cross_attention = False  # 单纯语言模型
        config.num_hidden_layers = 6  # 减少层数

        self.bert = BertModel.from_pretrained(pretrain_model_path, config=config)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.bert.embeddings.word_embeddings.weight  # tied weights
        self.lm_head.bias = nn.Parameter(torch.zeros(vocab_size))

        # 使用 Xavier 初始化
        nn.init.xavier_normal_(self.lm_head.weight)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        # 构造 causal mask（下三角）
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).unsqueeze(0).expand(
            batch_size, -1, -1)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=causal_mask  # 使用 causal mask
        )
        # 解构输出以获取 last_hidden_state
        hidden_states = outputs[0]  # (B, T, D)

        logits = self.lm_head(hidden_states)  # (B, T, V)

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        else:
            # 返回最后一个 token 的 logits（用于生成）
            return logits[:, -1, :].detach()


# ---------- 1. 加载 SFT 语料 ----------
def load_sft_dataset(json_path, tokenizer, max_len=64):
    """
    构造：[input][SEP][output][SEP]
    label: output 部分参与 loss 计算
    """
    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sep_token = "[SEP]"

    for item in raw:
        text = item["input"] + sep_token + item["output"] + sep_token
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False  # 动态 padding
        )
        input_ids = encoded["input_ids"]

        # 找到第一个 [SEP] 的位置（分隔 input 和 output）
        try:
            sep_id = tokenizer.convert_tokens_to_ids(sep_token)
            sep_positions = [i for i, x in enumerate(input_ids) if x == sep_id]
            if len(sep_positions) < 1:
                continue
            output_start = sep_positions[0] + 1  # output 从第一个 SEP 后开始
        except:
            output_start = len(input_ids)  # fallback

        # 构造 labels：只保留 output 部分，其余为 -100
        labels = [-100] * output_start + input_ids[output_start:]
        # 额外屏蔽 [SEP]
        labels = [tok if tok != sep_id else -100 for tok in labels]

        data.append({
            "input_ids": input_ids,
            "labels": labels
        })

    return data

# def load_sft_dataset(json_path, tokenizer, max_len=64):
#     """
#     构造：[input][SEP]主题：<主题>\n摘要：<摘要>[SEP]
#     其中：
#       - 前缀 "主题：" 与 "摘要：" 的 token 全部设为 -100，不参与 loss
#       - 真正需要学习的只有主题名 + 摘要句
#     """
#     data = []
#     sep_token = "[SEP]"
#     sep_id = tokenizer.convert_tokens_to_ids(sep_token)
#
#     with open(json_path, "r", encoding="utf-8") as f:
#         raw = json.load(f)
#
#     for item in raw:
#         # 1. 拼成完整文本
#         text = item["input"] + sep_token + item["output"] + sep_token
#
#         # 2. 编码
#         encoded = tokenizer(text, truncation=True, max_length=max_len, padding=False)
#         input_ids = encoded["input_ids"]
#
#         # 3. 找到 input 结束位置（即第一个 [SEP] 之后）
#         try:
#             sep_positions = [i for i, x in enumerate(input_ids) if x == sep_id]
#             if len(sep_positions) == 0:
#                 continue
#             output_start = sep_positions[0] + 1
#         except:
#             continue
#
#         # 4. 构造 labels：先全部屏蔽，再逐步放开
#         labels = [-100] * len(input_ids)
#
#         # 5. 把“主题：”和“摘要：”对应的 token 再次屏蔽
#         theme_prefix = tokenizer.encode("主题：", add_special_tokens=False)
#         summ_prefix  = tokenizer.encode("摘要：", add_special_tokens=False)
#
#         i = output_start
#         # 跳过“主题：”
#         if input_ids[i:i+len(theme_prefix)] == theme_prefix:
#             i += len(theme_prefix)
#         # 跳过“摘要：”
#         if input_ids[i:i+len(summ_prefix)] == summ_prefix:
#             i += len(summ_prefix)
#
#         # 6. 从 i 开始直到倒数第二个 token 参与 loss（最后一个 token 用于预测）
#         for j in range(i, len(input_ids) - 1):
#             labels[j] = input_ids[j + 1]
#
#         data.append({"input_ids": input_ids, "labels": labels})
#
#     return data

# ---------- 2. Dataset & Collate ----------
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    max_len = max(len(ids) for ids in input_ids)

    # 手动 padding
    padded_input_ids = []
    padded_labels = []
    for ids, lbs in zip(input_ids, labels):
        pad_num = max_len - len(ids)
        padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_num)
        padded_labels.append(lbs + [-100] * pad_num)

    return {
        "input_ids": torch.LongTensor(padded_input_ids),
        "labels": torch.LongTensor(padded_labels)
    }

def sample_top_p(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return torch.multinomial(softmax(logits, dim=-1), num_samples=1).item()

# ---------- 3. 训练 ----------
def train(json_path,
          pretrain_model_path=r"E:\BaiduNetdiskDownload\bert-base-chinese",
          save_dir="model",
          max_len=64,
          batch_size=16,
          epochs=50,  # 增加训练轮数
          lr=5e-5):  # 进一步降低学习率

    os.makedirs(save_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token or tokenizer.cls_token

    print("Loading dataset...")
    sft_data = load_sft_dataset(json_path, tokenizer, max_len)
    print(f"Loaded {len(sft_data)} samples.")

    dataset = SFTDataset(sft_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    model = CausalBertLM(vocab_size=tokenizer.vocab_size, pretrain_model_path=pretrain_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # 添加 L2 正则化
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率调度器

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, labels=labels)

            # 监控 loss 是否为 nan
            if torch.isnan(loss):
                print(f"NaN loss encountered at Epoch {epoch + 1}, Batch {batch_idx + 1}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 调整梯度裁剪阈值
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")

        # 更新学习率
        scheduler.step()

        # 保存模型
        model.bert.save_pretrained(f"{save_dir}/causal_bert_epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"{save_dir}/causal_bert_epoch_{epoch + 1}")

    print("Training completed.")
    return model, tokenizer


# ---------- 4. 生成测试 ----------
# def test_generate_sft(test_json_path,
#                       model_dir="model/causal_bert_epoch_50",  # 更新模型路径
#                       max_new_tokens=50,
#                       beam_width=5):  # 添加束宽参数
#     tokenizer = BertTokenizer.from_pretrained(model_dir)
#     config = BertConfig.from_pretrained(model_dir)
#     config.is_decoder = True
#     bert_model = BertModel.from_pretrained(model_dir, config=config)
#
#     model = CausalBertLM(vocab_size=tokenizer.vocab_size, pretrain_model_path=model_dir)
#     model.bert = bert_model  # 注入加载的 BERT
#     model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#
#     sep_token = "[SEP]"
#     sep_id = tokenizer.convert_tokens_to_ids(sep_token)
#     pad_token_id = tokenizer.pad_token_id
#
#     with open(test_json_path, "r", encoding="utf-8") as f:
#         test_samples = json.load(f)
#
#     for idx, sample in enumerate(test_samples, 1):
#         prompt = sample["input"]
#         golden = sample["output"].strip()
#
#         # 编码 prompt
#         prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
#         input_ids = torch.LongTensor([prompt_ids]).to(device)
#
#         # 自回归生成（使用束搜索）
#         sequences = [[list(input_ids[0]), 0.0]]  # 初始序列和累积概率
#
#         for _ in range(max_new_tokens):
#             all_candidates = []
#
#             for sequence, score in sequences:
#                 current_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
#                 seq_len = current_tensor.size(1)
#                 causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
#
#                 with torch.no_grad():
#                     logits = model(current_tensor).squeeze(0)  # shape: [vocab_size]
#
#                 log_probs = torch.log_softmax(logits, dim=-1)
#                 top_k_log_probs, top_k_indices = torch.topk(log_probs, k=beam_width, dim=-1)
#
#                 for j in range(beam_width):
#                     next_token = sample_top_p(logits, top_p=0.9, temperature=0.8)
#                     new_sequence = sequence + [next_token]
#                     new_score = score + top_k_log_probs[j].item()
#                     all_candidates.append((new_sequence, new_score))
#
#             # 选择最佳候选序列
#             sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
#
#             # 检查是否生成了结束符
#             for sequence, _ in sequences:
#                 if sequence[-1] == sep_id or sequence[-1] == pad_token_id:
#                     generated_ids = sequence
#                     break
#             else:
#                 generated_ids = sequences[0][0]
#
#         generated = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special_tokens=True).strip()
#
#         print(f"\n=========== 测试样例 {idx} ===========")
#         print("Prompt   :", prompt)
#         print("Golden   :", golden)
#         print("Generated:", generated)

def test_generate_sft(test_json_path,
                      model_dir="model/causal_bert_epoch_50",
                      max_new_tokens=50):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    config = BertConfig.from_pretrained(model_dir)
    config.is_decoder = True
    bert_model = BertModel.from_pretrained(model_dir, config=config)

    model = CausalBertLM(vocab_size=tokenizer.vocab_size, pretrain_model_path=model_dir)
    model.bert = bert_model
    model.lm_head.weight = model.bert.embeddings.word_embeddings.weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sep_token = "[SEP]"
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    pad_token_id = tokenizer.pad_token_id

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    for idx, sample in enumerate(test_samples, 1):
        prompt = sample["input"]
        golden = sample["output"].strip()

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = prompt_ids.copy()

        for _ in range(max_new_tokens):
            curr = torch.tensor(generated_ids).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(curr).squeeze(0)        # [vocab_size]
            next_token = sample_top_p(logits, top_p=0.9, temperature=0.8)
            generated_ids.append(next_token)
            if next_token in (sep_id, pad_token_id):
                break

        generated = tokenizer.decode(
            generated_ids[len(prompt_ids):], skip_special_tokens=True
        ).strip()

        print(f"\n=========== 测试样例 {idx} ===========")
        print("Prompt   :", prompt)
        print("Golden   :", golden)
        print("Generated:", generated)

# ========= 主程序 =========
if __name__ == "__main__":
    train_json = r"E:\BaiduNetdiskDownload\第十周\week10 文本生成问题\week10 文本生成问题\SFT\news_theme_sft.json"
    test_json = r"E:\BaiduNetdiskDownload\第十周\week10 文本生成问题\week10 文本生成问题\SFT\news_theme_test.json"

    # 训练
    trained_model, tokenizer = train(train_json)

    # 推理
    test_generate_sft(test_json, model_dir="model/causal_bert_epoch_50")



