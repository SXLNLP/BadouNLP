#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer

"""
基于BERT+mask 训练sft
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab, model_name='bert-base-chinese'):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab = vocab
        
        self.classify = nn.Linear(self.bert.config.hidden_size, self.tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy
        
        # 添加mask token
        self.mask_token_id = self.tokenizer.mask_token_id
        
        print(f"模型初始化完成：使用{self.bert.config.num_hidden_layers}层BERT")

    def forward(self, input_ids, attention_mask=None, masked_positions=None, y=None, loss_mask=None):
        # 根据参考代码实现因果mask
        if loss_mask is not None:  # SFT训练模式
            # 构建下三角mask矩阵，让模型只能看到前面的内容
            causal_mask = torch.tril(torch.ones((input_ids.shape[0], input_ids.shape[1], input_ids.shape[1])))
            if torch.cuda.is_available():
                causal_mask = causal_mask.cuda()
            
            # 使用因果mask进行BERT编码
            outputs = self.bert(input_ids=input_ids, attention_mask=causal_mask)
        else:
            # 原始mask语言模型模式或预测模式
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # 分类层
        y_pred = self.classify(sequence_output)  # (batch_size, seq_len, vocab_size)
        
        if y is not None:
            if loss_mask is not None:
                # SFT训练模式：标准的语言模型训练方式
                # y_pred: (batch_size, seq_len, vocab_size)
                # y: (batch_size, seq_len) - 目标tokens
                # loss_mask: (batch_size, seq_len) - 只对response部分计算loss
                
                # 展平所有维度
                flat_logits = y_pred.view(-1, y_pred.size(-1))  # (batch_size * seq_len, vocab_size)
                flat_labels = y.view(-1)  # (batch_size * seq_len,)
                flat_loss_mask = loss_mask.view(-1)  # (batch_size * seq_len,)
                
                # 只计算loss_mask为1的位置
                if flat_loss_mask.sum() > 0:
                    # 选择需要计算loss的位置
                    valid_indices = flat_loss_mask.bool()
                    valid_logits = flat_logits[valid_indices]
                    valid_labels = flat_labels[valid_indices]
                    
                    # 过滤掉padding tokens（通常是0）
                    non_pad_mask = valid_labels != self.tokenizer.pad_token_id
                    if non_pad_mask.sum() > 0:
                        final_logits = valid_logits[non_pad_mask]
                        final_labels = valid_labels[non_pad_mask]
                        return self.loss(final_logits, final_labels)
                    
                # 如果没有有效的训练位置，返回0
                return torch.tensor(0.0, requires_grad=True, device=input_ids.device)
            elif masked_positions is not None:
                # 原始mask语言模型训练模式
                batch_indices, pos_indices = masked_positions
                masked_y_pred = y_pred[batch_indices, pos_indices]
                masked_y = y
                return self.loss(masked_y_pred, masked_y)
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#加载SFT训练数据
def load_sft_data(path):
    import json
    sft_data = []
    with open(path, encoding="utf8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                # 替换中文引号为英文引号
                line = line.replace('"', '"').replace('"', '"')
                data = json.loads(line)
                # 构造指令-回答对
                prompt = f"标题：{data['title']}\n内容："
                response = data['content']
                sft_data.append({"prompt": prompt, "response": response})
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败: {e}")
                print(f"问题行内容: {line[:100]}...")
                continue
    return sft_data

# SFT训练样本构建：将prompt和response拼接，只对response部分计算loss
def build_sft_sample(tokenizer, sft_item, max_length):
    prompt = sft_item["prompt"]
    response = sft_item["response"]
    
    # 编码prompt和response
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    response_tokens = tokenizer.encode(response, add_special_tokens=False)
    
    # 构建完整序列：[CLS] + prompt + [SEP] + response + [SEP] 
    # 这样更符合BERT的输入格式，两个句子用[SEP]分隔
    full_tokens = [tokenizer.cls_token_id] + prompt_tokens + [tokenizer.sep_token_id] + response_tokens + [tokenizer.sep_token_id]
    
    # 截断处理
    if len(full_tokens) > max_length:
        # 优先保留prompt，截断response
        prompt_part_length = len(prompt_tokens) + 3  # [CLS] + prompt + [SEP] + [SEP]
        available_length = max_length - prompt_part_length
        if available_length > 0:
            response_tokens = response_tokens[:available_length]
        else:
            # 如果prompt太长，也要截断
            prompt_tokens = prompt_tokens[:max_length-3]
            response_tokens = []
        
        full_tokens = [tokenizer.cls_token_id] + prompt_tokens + [tokenizer.sep_token_id] + response_tokens + [tokenizer.sep_token_id]
    
    input_ids = full_tokens[:-1]  # 去掉最后一个[SEP]作为输入
    targets = full_tokens[1:]     # 从第二个token开始作为目标
    
    # loss_mask: 只对response部分计算loss
    loss_mask = [0] * len(input_ids)
    
    if len(response_tokens) > 0:
        # response在input_ids中的起始位置（考虑[CLS] + prompt + [SEP]）
        response_start_in_input = 1 + len(prompt_tokens) + 1  # [CLS] + prompt + [SEP]
        response_end_in_input = response_start_in_input + len(response_tokens)
        
        # 对response部分设置loss_mask为1
        for i in range(response_start_in_input, min(response_end_in_input, len(loss_mask))):
            loss_mask[i] = 1
    
    return input_ids, targets, loss_mask

# 改进的mask策略：随机mask部分token用于语言建模
def build_sample(tokenizer, vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    
    # 使用BERT tokenizer编码
    tokens = tokenizer.encode(window, add_special_tokens=True, max_length=window_size+2, truncation=True)
    
    # 如果tokens太短，跳过
    if len(tokens) < 5:
        return build_sample(tokenizer, vocab, window_size, corpus)
    
    # 随机mask 15%的token（类似BERT预训练）
    input_ids = tokens.copy()
    masked_positions = []
    targets = []
    
    # 不mask [CLS]和[SEP]
    maskable_positions = list(range(1, len(tokens) - 1))
    num_to_mask = max(1, int(len(maskable_positions) * 0.15))
    
    mask_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
    
    for pos in mask_positions:
        targets.append(input_ids[pos])
        input_ids[pos] = tokenizer.mask_token_id
        masked_positions.append(pos)
    
    # 如果没有mask任何位置，重新生成
    if not targets:
        return build_sample(tokenizer, vocab, window_size, corpus)
    
    return input_ids, targets, masked_positions

# SFT数据集构建
def build_sft_dataset(sample_length, tokenizer, sft_data, max_length):
    dataset_input_ids = []
    dataset_targets = []
    dataset_loss_masks = []
    
    # 随机采样SFT数据
    sampled_data = random.choices(sft_data, k=sample_length)
    
    max_len = 0
    for sft_item in sampled_data:
        input_ids, targets, loss_mask = build_sft_sample(tokenizer, sft_item, max_length)
        dataset_input_ids.append(input_ids)
        dataset_targets.append(targets)
        dataset_loss_masks.append(loss_mask)
        max_len = max(max_len, len(input_ids))
    
    # Padding
    padded_input_ids = []
    attention_masks = []
    padded_targets = []
    padded_loss_masks = []
    
    for i in range(sample_length):
        input_ids = dataset_input_ids[i]
        targets = dataset_targets[i]
        loss_mask = dataset_loss_masks[i]
        
        # Padding
        pad_len = max_len - len(input_ids)
        padded_input = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        padded_target = targets + [tokenizer.pad_token_id] * pad_len
        padded_loss_mask = loss_mask + [0] * pad_len  # padding部分不计算loss
        
        padded_input_ids.append(padded_input)
        attention_masks.append(attention_mask)
        padded_targets.append(padded_target)
        padded_loss_masks.append(padded_loss_mask)
    
    return (torch.LongTensor(padded_input_ids), 
            torch.LongTensor(attention_masks),
            torch.LongTensor(padded_targets), 
            torch.LongTensor(padded_loss_masks))

def build_dataset(sample_length, tokenizer, vocab, window_size, corpus):
    dataset_input_ids = []
    dataset_targets = []
    dataset_masked_positions = []
    
    max_len = 0
    for i in range(sample_length):
        input_ids, targets, masked_positions = build_sample(tokenizer, vocab, window_size, corpus)
        dataset_input_ids.append(input_ids)
        dataset_targets.append(targets)
        dataset_masked_positions.append(masked_positions)
        max_len = max(max_len, len(input_ids))
    
    # Padding
    padded_input_ids = []
    attention_masks = []
    padded_targets = []
    padded_masked_positions = []
    
    for i in range(sample_length):
        input_ids = dataset_input_ids[i]
        targets = dataset_targets[i]
        masked_positions = dataset_masked_positions[i]
        
        # Padding input_ids
        pad_len = max_len - len(input_ids)
        padded_input = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        
        padded_input_ids.append(padded_input)
        attention_masks.append(attention_mask)
        padded_targets.append(targets)
        padded_masked_positions.append(masked_positions)
    
    return (torch.LongTensor(padded_input_ids), 
            torch.LongTensor(attention_masks),
            padded_targets, 
            padded_masked_positions)

def build_model(vocab):
    model = LanguageModel(vocab)
    return model

def generate_sentence(openings, model, max_length):
    tokenizer = model.tokenizer
    model.eval()
    with torch.no_grad():
        # 编码输入prompt
        input_text = openings
        input_tokens = tokenizer.encode(input_text, add_special_tokens=True, max_length=max_length//2, truncation=True)
        
        generated_tokens = input_tokens.copy()
        
        for _ in range(50):  # 最多生成50个token
            # 创建输入
            input_ids = torch.LongTensor([generated_tokens])
            attention_mask = torch.ones_like(input_ids)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            # 获取模型输出 - 使用因果mask进行生成
            # 构建因果mask用于生成（参考bert_nnlm.py的实现）
            causal_mask = torch.tril(torch.ones((input_ids.shape[0], input_ids.shape[1], input_ids.shape[1])))
            if torch.cuda.is_available():
                causal_mask = causal_mask.cuda()
            
            # 使用因果mask获取输出
            bert_outputs = model.bert(input_ids=input_ids, attention_mask=causal_mask)
            sequence_output = bert_outputs[0]
            logits = model.classify(sequence_output)
            outputs = torch.softmax(logits, dim=-1)
            
            # 获取最后一个位置的预测
            last_token_logits = logits[0, -1, :]  # 最后一个位置的logits用于采样
            
            # 采样策略 - 使用保守的采样避免奇怪字符
            debug_sampling = len(generated_tokens) <= 5  # 只对前几个token调试
            next_token_id = top_k_sampling(last_token_logits, tokenizer=tokenizer, debug=debug_sampling)
            
            # 检查是否为特殊token
            if next_token_id in [tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.unk_token_id]:
                break
            
            # 添加到生成序列
            generated_tokens.append(next_token_id)
            
            # 解码当前token检查结束条件
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            if next_token in ['。', '！', '？']:
                break
            
            # 避免序列过长
            if len(generated_tokens) > max_length:
                break
        
        # 解码完整生成的文本，并处理空格问题
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 移除中文字符间的空格
        import re
        # 移除中文字符间的空格，但保留必要的标点空格
        generated_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', generated_text)
        generated_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[，。！？：；])', '', generated_text)
        generated_text = re.sub(r'(?<=[，。！？：；])\s+(?=[\u4e00-\u9fff])', '', generated_text)
        
        return generated_text

def top_k_sampling(logits, k=3, temperature=0.5, tokenizer=None, debug=False):
    """保守的采样策略，避免生成奇怪字符"""
    # 应用更低的温度，增加确定性
    logits = logits / temperature
    
    # 更保守的中文字符范围
    # 根据常用字符分布，大部分中文在这个范围
    chinese_start = 1000
    chinese_end = 8000
    
    # 创建更严格的限制
    allowed_logits = torch.full_like(logits, float('-inf'))
    
    # 只允许中文字符范围
    if len(logits) > chinese_end:
        allowed_logits[chinese_start:chinese_end] = logits[chinese_start:chinese_end]
    
    # 也允许一些基础符号（句号、逗号等）
    basic_punctuation = [102, 511, 8043, 8024]  # [SEP], 。, ，等
    for idx in basic_punctuation:
        if idx < len(logits):
            allowed_logits[idx] = logits[idx]
    
    # 获取top-k
    top_k_logits, top_k_indices = torch.topk(allowed_logits, k)
    
    if debug and tokenizer:
        print("采样候选:")
        for i, (logit, idx) in enumerate(zip(top_k_logits, top_k_indices)):
            if idx < len(tokenizer.vocab):
                token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
                prob = torch.softmax(top_k_logits, dim=-1)[i].item()
                print(f"  {i+1}. Token: '{token}' (ID: {idx.item()}, Prob: {prob:.3f})")
    
    # 简单选择概率最高的
    return top_k_indices[0].item()

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def calc_perplexity(sentence, model, window_size):
    tokenizer = model.tokenizer
    vocab = model.vocab
    prob = 0
    model.eval()
    
    with torch.no_grad():
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        
        for i in range(1, len(tokens) - 1):  # 跳过[CLS]和[SEP]
            # 创建输入，将当前位置mask
            input_tokens = tokens[:i] + [tokenizer.mask_token_id] + tokens[i+1:]
            input_ids = torch.LongTensor([input_tokens])
            attention_mask = torch.ones_like(input_ids)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            # 预测
            outputs = model(input_ids, attention_mask)
            pred_probs = outputs[0, i, :]  # 被mask位置的预测
            
            # 计算目标token的概率
            target_token_id = tokens[i]
            if target_token_id < len(pred_probs):
                target_prob = torch.softmax(pred_probs, dim=0)[target_token_id]
                prob += math.log(target_prob.item(), 10)
    
    return 2 ** (prob * (-1 / len(tokens)))


def train_sft(sft_data_path, save_weight=True):
    """SFT训练函数"""
    epoch_num = 12        # 适中的epoch数
    batch_size = 4        # 适中的batch_size
    train_sample = 800    # 增加训练样本数
    max_length = 96       # 适中的序列长度
    vocab = build_vocab("bert-base-chinese/vocab.txt")
    sft_data = load_sft_data(sft_data_path)
    
    # 验证数据质量
    if len(sft_data) == 0:
        print("错误：没有加载到任何SFT数据！")
        return
    
    print(f"成功加载 {len(sft_data)} 条SFT训练数据")
    print("数据样本:")
    for i in range(min(3, len(sft_data))):
        print(f"  样本{i+1} - Prompt: {sft_data[i]['prompt'][:40]}...")
        print(f"           Response: {sft_data[i]['response'][:40]}...")
    
    model = build_model(vocab)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 使用更低的学习率和更温和的学习率调度
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=8, gamma=0.8)  # 更温和的衰减

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for batch in range(int(train_sample / batch_size)):
            input_ids, attention_masks, targets, loss_masks = build_sft_dataset(
                batch_size, model.tokenizer, sft_data, max_length
            )
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_masks = attention_masks.cuda()
                targets = targets.cuda()
                loss_masks = loss_masks.cuda()
            
            # 计算loss
            loss = model(input_ids, attention_masks, y=targets, loss_mask=loss_masks)
            
            # 调试信息
            if batch == 0 and epoch == 0:
                print(f"调试信息 - Batch {batch}:")
                print(f"  input_ids shape: {input_ids.shape}")
                print(f"  targets shape: {targets.shape}")
                print(f"  loss_masks shape: {loss_masks.shape}")
                print(f"  loss_masks sum: {loss_masks.sum().item()}")
                print(f"  使用因果mask: True (下三角矩阵)")
                print(f"  loss value: {loss.item():.4f}")
            
            if loss.item() > 0:  # 只有当loss大于0时才进行反向传播
                optim.zero_grad()
                loss.backward()
                optim.step()
                watch_loss.append(loss.item())
        
        # 每轮结束后更新学习率
        scheduler.step()
        
        if watch_loss:
            print("=========\n第%d轮平均loss:%f, 学习率:%.2e" % (epoch + 1, np.mean(watch_loss), optim.param_groups[0]['lr']))
            # 测试生成 - 使用训练数据中的真实格式
            if len(sft_data) > 0:
                sample_item = sft_data[0]
                test_prompt = sample_item["prompt"]
                print(f"测试prompt: {test_prompt[:30]}...")
                generated = generate_sentence(test_prompt, model, max_length)
                print("测试生成：", generated[:100])
                print(f"预期response: {sample_item['response'][:30]}...")
    
    if save_weight:
        model_path = os.path.join("model", "sft_model.pth")
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")




if __name__ == "__main__":
    # SFT训练模式
    train_sft("sample_data.json", True)
