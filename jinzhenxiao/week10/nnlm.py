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
基于BERT+mask的自回归语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab, model_name='bert-base-chinese'):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab = vocab
        # 使用BERT的vocab大小而不是自定义vocab
        self.classify = nn.Linear(self.bert.config.hidden_size, self.tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy
        
        # 添加mask token
        self.mask_token_id = self.tokenizer.mask_token_id

    def forward(self, input_ids, attention_mask=None, masked_positions=None, y=None):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # 分类层
        y_pred = self.classify(sequence_output)  # (batch_size, seq_len, vocab_size)
        
        if y is not None and masked_positions is not None:
            # 只计算被mask位置的loss
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

def generate_sentence(openings, model, window_size):
    tokenizer = model.tokenizer
    model.eval()
    with torch.no_grad():
        generated_text = openings
        
        for _ in range(20):  # 最多生成20个token
            # 编码当前文本并添加一个mask token用于预测下一个词
            current_tokens = tokenizer.encode(generated_text, add_special_tokens=False, max_length=window_size-2, truncation=True)
            
            # 构建输入：[CLS] + current_tokens + [MASK] + [SEP]
            input_tokens = [tokenizer.cls_token_id] + current_tokens + [tokenizer.mask_token_id] + [tokenizer.sep_token_id]
            
            # 创建输入
            input_ids = torch.LongTensor([input_tokens])
            attention_mask = torch.ones_like(input_ids)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            # 预测mask位置的token
            outputs = model(input_ids, attention_mask)
            mask_pos = len(current_tokens) + 1  # mask token的位置
            pred_probs = outputs[0, mask_pos, :]  # mask位置的预测
            
            # 采样策略
            next_token_id = sampling_strategy(pred_probs)
            
            # 解码并检查结束条件
            if next_token_id in [tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id]:
                break
                
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            
            if not next_token or next_token.strip() == '':
                break
                
            generated_text += next_token
            
            # 如果生成了句号或换行，停止生成
            if next_token in ['。', '！', '？', '\n']:
                break
                
    return generated_text

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


def train(corpus_path, save_weight=True):
    epoch_num = 20        
    batch_size = 16       # 增加batch_size提高梯度稳定性
    train_sample = 10000  # 增加样本数量
    window_size = 32      # 减小窗口大小，适合中文
    vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    model = build_model(vocab)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 增加学习 率，加快收敛
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    print("文本词表模型加载完毕，开始训练")
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for batch in range(int(train_sample / batch_size)):
            input_ids, attention_masks, targets_list, masked_positions_list = build_dataset(
                batch_size, model.tokenizer, vocab, window_size, corpus
            )
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_masks = attention_masks.cuda()
            
            total_loss = 0
            valid_samples = 0
            
            # 处理每个样本
            for i in range(batch_size):
                if len(targets_list[i]) > 0 and len(masked_positions_list[i]) > 0:
                    # 创建mask positions tensor
                    masked_pos = masked_positions_list[i]
                    targets = torch.LongTensor(targets_list[i])
                    
                    # 检查position是否超出序列长度
                    seq_len = input_ids[i].size(0)
                    valid_positions = [pos for pos in masked_pos if pos < seq_len]
                    
                    if not valid_positions:
                        continue
                    
                    # 只使用有效的位置
                    valid_targets = targets[:len(valid_positions)]
                    
                    if torch.cuda.is_available():
                        valid_targets = valid_targets.cuda()
                    
                    # 创建二维mask positions - 使用单个样本的索引
                    batch_indices = torch.zeros(len(valid_positions), dtype=torch.long)
                    pos_indices = torch.LongTensor(valid_positions)
                    
                    if torch.cuda.is_available():
                        batch_indices = batch_indices.cuda()
                        pos_indices = pos_indices.cuda()
                    
                    masked_positions = (batch_indices, pos_indices)
                    
                    # 计算loss
                    loss = model(input_ids[i:i+1], attention_masks[i:i+1], masked_positions, valid_targets)
                    total_loss += loss
                    valid_samples += 1
            
            if valid_samples > 0:
                avg_loss = total_loss / valid_samples
                optim.zero_grad()
                avg_loss.backward()
                optim.step()
                watch_loss.append(avg_loss.item())
        
        if watch_loss:
            print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
            print(generate_sentence("让他在半年之前", model, window_size))
            print(generate_sentence("李慕站在山路上", model, window_size))
    
    if save_weight:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    train("corpus.txt", True)
