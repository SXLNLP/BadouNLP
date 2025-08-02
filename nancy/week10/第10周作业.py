#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import  BertTokenizer, BertForMaskedLM

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""
基于BERT+mask的自回归语言模型
"""

class BertLanguageModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese'):
        super(BertLanguageModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertForMaskedLM.from_pretrained(bert_model_name)
        self.mask_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, labels=None):
        if labels is not None:
            outputs = self.model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels)
            return outputs[0]
        else:
            outputs = self.model(input_ids=input_ids, 
                                attention_mask=attention_mask)
            return torch.softmax(outputs[0], dim=-1)

# 加载BERT词表
def build_vocab(bert_model_name='bert-base-chinese'):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    return tokenizer

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 随机生成一个样本
# 从文本中截取随机窗口，随机mask一个位置作为预测目标
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - window_size)
    window = corpus[start:start + window_size]
    
    # 分词
    tokens = tokenizer.tokenize(window)
    if len(tokens) <= 2:  # 确保有足够的token可以mask
        return build_sample(tokenizer, window_size, corpus)
    
    # 随机选择一个位置进行mask
    mask_pos = random.randint(0, len(tokens) - 1)
    original_token = tokens[mask_pos]
    tokens[mask_pos] = tokenizer.mask_token
    
    # 转换为ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 创建标签，只在mask位置有值，其他位置为-100（忽略）
    labels = [-100] * len(input_ids)
    labels[mask_pos] = tokenizer.convert_tokens_to_ids([original_token])[0]
    
    return input_ids, labels, mask_pos

# 建立数据集
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_inputs = []
    dataset_labels = []
    dataset_mask_pos = []
    
    for i in range(sample_length):
        input_ids, labels, mask_pos = build_sample(tokenizer, window_size, corpus)
        dataset_inputs.append(input_ids)
        dataset_labels.append(labels)
        dataset_mask_pos.append(mask_pos)
    
    # 填充处理
    max_len = max([len(x) for x in dataset_inputs])
    for i in range(len(dataset_inputs)):
        padding_length = max_len - len(dataset_inputs[i])
        dataset_inputs[i] = dataset_inputs[i] + [tokenizer.pad_token_id] * padding_length
        dataset_labels[i] = dataset_labels[i] + [-100] * padding_length
    
    return (torch.LongTensor(dataset_inputs), 
            torch.LongTensor(dataset_labels),
            torch.LongTensor(dataset_mask_pos))

# 建立模型
def build_model(bert_model_name='bert-base-chinese'):
    model = BertLanguageModel(bert_model_name)
    return model

# 文本生成测试代码 - 自回归方式
def generate_sentence(openings, model, tokenizer, max_length=30):
    model.model.eval()
    with torch.no_grad():
        generated_text = openings
        
        # 自回归生成
        for _ in range(max_length):
            # 对当前文本进行tokenize
            tokens = tokenizer.tokenize(generated_text)
            if len(tokens) > 512 - 2:  # BERT最大长度限制
                tokens = tokens[-(512-2):]
            
            # 在末尾添加MASK标记
            tokens.append(tokenizer.mask_token)
            
            # 转换为ID
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.LongTensor([input_ids])
            
            input_ids = input_ids.to(device)
            
            # 预测下一个token
            outputs = model.model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits  # 兼容新旧版本
            predictions = torch.softmax(logits[0, -1], dim=-1)
            
            # 采样下一个token
            next_token_id = sampling_strategy(predictions)
            next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]
            
            # 如果生成了[SEP]或特殊标记，则停止
            if next_token == '[SEP]' or next_token in ['[CLS]', '[PAD]']:
                break
                
            # 添加到生成文本中
            if next_token.startswith('##'):
                generated_text += next_token[2:]
            else:
                generated_text += ' ' + next_token
                
            # 清理文本（针对中文）
            generated_text = generated_text.replace(' ', '')
            
            # 如果生成了换行符，则停止
            if '\n' in next_token:
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

# 计算文本ppl
def calc_perplexity(sentence, model, tokenizer, window_size=50):
    model.model.eval()
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) <= 1:
        return float('inf')
    
    log_likelihood = 0.0
    count = 0
    
    with torch.no_grad():
        for i in range(len(tokens)):
            # 复制tokens并mask当前位置
            masked_tokens = tokens.copy()
            if i < len(masked_tokens):
                original_token = masked_tokens[i]
                masked_tokens[i] = tokenizer.mask_token
                
                # 转换为ID
                input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
                input_ids = torch.LongTensor([input_ids])
                
                input_ids = input_ids.to(device)
                
                # 获取预测
                outputs = model.model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                predictions = torch.softmax(logits[0, i], dim=-1)
                
                # 获取原始token的概率
                original_id = tokenizer.convert_tokens_to_ids([original_token])[0]
                token_prob = predictions[original_id].item()
                
                if token_prob > 0:
                    log_likelihood += math.log(token_prob)
                    count += 1
    
    if count == 0:
        return float('inf')
    
    ppl = math.exp(-log_likelihood / count)
    return ppl

def train(corpus_path, save_weight=True, bert_model_name='bert-base-chinese'):
    epoch_num = 20        #训练轮数
    batch_size = 16        #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    window_size = 10      #样本文本长度
    
    tokenizer = build_vocab(bert_model_name)
    corpus = load_corpus(corpus_path)
    model = build_model(bert_model_name)
    
    model.model = model.model.to(device)
    optim = torch.optim.AdamW(model.model.parameters(), lr=5e-5)
    
    print("BERT模型和词表加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.model.train()
        watch_loss = []
        
        for batch in range(int(train_sample / batch_size)):
            inputs, labels, _ = build_dataset(batch_size, tokenizer, window_size, corpus)
            
            # 创建attention mask
            attention_mask = (inputs != tokenizer.pad_token_id).long()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            optim.zero_grad()
            loss = model(inputs, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optim.step()
            
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))
    
    if save_weight:
        base_name = os.path.basename(corpus_path).replace(".txt", "")
        model_path = os.path.join("model", f"bert_{base_name}")
        os.makedirs(model_path, exist_ok=True)
        model.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    
    return

if __name__ == "__main__":
    train("corpus.txt", True)
