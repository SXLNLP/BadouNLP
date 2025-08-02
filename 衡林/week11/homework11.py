# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class NewsDataset(Dataset):
    def __init__(self, corpus_data, tokenizer, max_length):
        self.corpus_data = corpus_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.corpus_data)
    
    def __getitem__(self, idx):
        item = self.corpus_data[idx]
        title = item.get("title", "")
        content = item.get("content", "")
        
        text = f"标题：{title}[SEP]内容：{content}"
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # 创建标签（预测下一个token）
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:].clone()
        labels[-1] = -100  # 忽略最后一个token的预测
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # 加载配置以获取特殊token id
        config = BertConfig.from_pretrained(pretrain_model_path)
        self.bert = BertModel.from_pretrained(pretrain_model_path, config=config)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.sep_token_id = config.sep_token_id
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 创建因果注意力掩码
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        
        # 创建扩展的注意力掩码 (batch_size, 1, seq_len, seq_len)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        
        # 找到分隔符位置
        sep_positions = (input_ids == self.sep_token_id).nonzero(as_tuple=True)[1]
        if sep_positions.numel() == 0:
            sep_positions = torch.full((batch_size,), seq_len-1, device=device)
        
        # 创建因果掩码
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len)
        
        # 标题部分可以看到全部标题内容
        for i in range(batch_size):
            sep_pos = sep_positions[i] if sep_positions.numel() > 1 else sep_positions
            causal_mask[i, :, :sep_pos+1, :sep_pos+1] = 1
        
        # 合并注意力掩码
        extended_attention_mask = extended_attention_mask * causal_mask
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=extended_attention_mask.squeeze(1),
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )
        
        sequence_output = outputs[0]
        logits = self.classify(sequence_output)
        
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        return logits


def load_json_corpus(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 尝试读取为JSON数组
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                # 如果不是标准JSON，尝试逐行读取
                f.seek(0)
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError:
                            continue
                return data
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return []


def generate_sentence(prompt, model, tokenizer, max_len=256, temperature=0.9, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_len):
            # 限制输入长度
            inputs = generated[:, -max_len:] if generated.size(1) > max_len else generated
            
            # 获取模型输出
            logits = model(inputs)
            next_token_logits = logits[0, -1, :]
            
            # 应用温度采样和top-k筛选
            next_token_logits = next_token_logits / temperature
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # 采样
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[torch.multinomial(probs, num_samples=1)]
            
            # 如果生成了[SEP]或者[PAD]，则停止
            if next_token.item() in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                break
                
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def train(corpus_path, save_weight=True):
    # 超参数配置
    config = {
        "epoch_num": 20,
        "batch_size": 32,
        "max_length": 128,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "save_dir": "model",
        "pretrain_model_path": "bert-base-chinese"
    }
    
    # 确保保存目录存在
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
    corpus_data = load_json_corpus(corpus_path)
    
    if not corpus_data:
        raise ValueError("No valid data loaded from JSON file")
    
    # 创建数据集和数据加载器
    dataset = NewsDataset(corpus_data, tokenizer, config["max_length"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    # 初始化模型
    model = LanguageModel(
        hidden_size=768,
        vocab_size=len(tokenizer),
        pretrain_model_path=config["pretrain_model_path"]
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 设置优化器和学习率调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["learning_rate"])
    total_steps = len(dataloader) * config["epoch_num"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    print("开始训练...")
    for epoch in range(config["epoch_num"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # 生成示例
        print("\n生成示例:")
        prompts = [
            "标题：世界杯决赛",
            "标题：科技公司发布新产品",
            "标题：金融市场动态"
        ]
        
        for prompt in prompts:
            generated = generate_sentence(prompt, model, tokenizer)
            print(f"输入: {prompt}")
            print(f"生成: {generated}\n")
        
        # 保存模型
        if save_weight and (epoch + 1) % 5 == 0:
            model_path = os.path.join(
                config["save_dir"],
                f"news_generator_epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")


if __name__ == "__main__":
    train("sample_data.json", save_weight=True)
