import torch
import torch.nn as nn
import numpy as np
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import json
from torch.utils.data import Dataset, DataLoader
import random



DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')    
BERT_PATH = '/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese'
TRAIN_FILE = 'data/train_triplet.json'
VALID_FILE = 'data/valid_triplet.json'
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5


tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)
bert = bert.to(DEVICE)
bert.train()

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(bert.parameters(), lr=2e-5)

class TripletTextDataset(Dataset):
    def __init__(self, json_file):
        self.samples = []
        with open(json_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if all(k in obj for k in ['anchor', 'positive', 'negative']):
                            self.samples.append(obj)
                        else:
                            print(f"第{i}行缺少字段: {obj}")
                    except Exception as e:
                        print(f"第{i}行解析失败: {line.strip()}，错误：{e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item['anchor'], item['positive'], item['negative']

def get_dataloader(json_file, batch_size=16, shuffle=True):
    dataset = TripletTextDataset(json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_cls_embedding(texts, tokenizer, bert, device=DEVICE):
    inputs = tokenizer(list(texts), return_tensors='pt', truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert(**inputs)
    if isinstance(outputs, tuple):
        last_hidden_state = outputs[0]
    else:
        last_hidden_state = outputs.last_hidden_state
    return last_hidden_state[:, 0, :]

def evaluate(dataloader):
    bert.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:  
            anchor, positive, negative = batch
            anchor_vec = get_cls_embedding(anchor, tokenizer, bert)
            positive_vec = get_cls_embedding(positive, tokenizer, bert)
            negative_vec = get_cls_embedding(negative, tokenizer, bert)
            loss = triplet_loss(anchor_vec, positive_vec, negative_vec)
            total_loss += loss.item()
            count += 1
    bert.train()
    return total_loss / count if count > 0 else 0

def train():
    train_loader = get_dataloader(TRAIN_FILE, batch_size=BATCH_SIZE)
    valid_loader = get_dataloader(VALID_FILE, batch_size=BATCH_SIZE, shuffle=False)
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:  
            anchor, positive, negative = batch
            anchor_vec = get_cls_embedding(anchor, tokenizer, bert)
            positive_vec = get_cls_embedding(positive, tokenizer, bert)
            negative_vec = get_cls_embedding(negative, tokenizer, bert)
            loss = triplet_loss(anchor_vec, positive_vec, negative_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss = evaluate(valid_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")


import json
import random

input_file = 'data/valid.json'
output_file = 'data/valid_triplet.json'

# 读取所有样本，每行为 [question, label]
with open(input_file, encoding='utf-8') as f:
    lines = [json.loads(line) for line in f if line.strip()]

# 按标签分组
label2questions = {}
for item in lines:
    if isinstance(item, list) and len(item) == 2:
        question, label = item
        label2questions.setdefault(label, []).append(question)

# 构建三元组
triplets = []
labels = list(label2questions.keys())
for label, questions in label2questions.items():
    if len(questions) < 2:
        continue
    for i in range(len(questions) - 1):
        anchor = questions[i]
        positive = questions[i + 1]
        # 选一个不同标签的负样本
        negative_label = random.choice([l for l in labels if l != label])
        negative = random.choice(label2questions[negative_label])
        triplets.append({'anchor': anchor, 'positive': positive, 'negative': negative})

# 写入新文件
with open(output_file, 'w', encoding='utf-8') as f:
    for triplet in triplets:
        f.write(json.dumps(triplet, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    train()

        
