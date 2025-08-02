# F:\LeetCoding\torch\.idea\week10 文本生成问题\lstm语言模型生成文本\week10Work.py
#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
import re
import json
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

"""
基于pytorch和transformers库的BERT问答对监督微调模型
"""

class BERTLanguageModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super(BERTLanguageModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)
        self.model = BertForMaskedLM.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

def load_samples(json_path):
    """加载JSON格式的问答数据"""
    samples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append((data['title'], data['content']))
    return samples

def build_dataset(samples, tokenizer, max_len=128):
    """构建问答对数据集"""
    input_ids_list = []
    attention_masks = []
    labels_list = []
    
    for question, answer in samples:
        # 编码输入输出
        encoded_input = tokenizer.encode_plus(
            question, 
            add_special_tokens=True,
            max_length=max_len//2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoded_output = tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=max_len//2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 拼接输入输出
        input_ids = torch.cat([encoded_input['input_ids'], encoded_output['input_ids']], dim=1)[0]
        attention_mask = torch.cat([encoded_input['attention_mask'], encoded_output['attention_mask']], dim=1)[0]
        
        # 构造标签：仅计算输出部分的loss
        labels = torch.cat([
            torch.tensor([-100] * encoded_input['input_ids'].size(1)),  # 忽略输入部分
            encoded_output['input_ids'][0]  # 计算输出部分loss
        ])
        
        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        labels_list.append(labels)
    
    return (
        torch.stack(input_ids_list),
        torch.stack(attention_masks),
        torch.stack(labels_list)
    )

def build_model(model_name='bert-base-chinese'):
    model = BERTLanguageModel(model_name)
    return model

def train(corpus_path, save_weight=True):
    epoch_num = 5         # 训练轮数
    batch_size = 8        # 每次训练样本个数
    window_size = 128     # 样本文本长度

    # 构建完整的文件路径
    base_dir = os.path.dirname(corpus_path)
    
    # 加载JSON数据
    samples = load_samples(corpus_path)
    tokenizer = BertTokenizer.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)
    input_ids, attention_mask, labels = build_dataset(samples, tokenizer, window_size)
    
    # 建立模型
    model = build_model()
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=5e-5)  # BERT推荐学习率
    print("BERT监督微调模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 按batch训练
        for i in range(0, len(input_ids), batch_size):
            batch_input = input_ids[i:i+batch_size]
            batch_mask = attention_mask[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            if torch.cuda.is_available():
                batch_input, batch_mask = batch_input.cuda(), batch_mask.cuda()
                batch_labels = batch_labels.cuda()

            optim.zero_grad()
            outputs = model(
                input_ids=batch_input, 
                attention_mask=batch_mask, 
                labels=batch_labels
            )
            
            # 修正输出处理逻辑
            if isinstance(outputs, tuple):
                # 当模型输出为元组时，第一个元素是loss
                loss = outputs[0]
            else:
                # 当模型输出为ModelOutput对象时，使用loss属性
                loss = outputs.loss
                
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

    if save_weight:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", "sft_" + base_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    # 使用完整路径
    current_dir = "F:/BaiduNetdiskDownload/八斗精品班/第十周/week10 文本生成问题/transformers-生成文章标题/"
    corpus_path = os.path.join(current_dir, "sample_data.json")
    train(corpus_path, False)
