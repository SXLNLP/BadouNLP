import torch
import torch.nn as nn
import numpy as np
import json
import os
import re
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 建立模型
def build_model(bert_model_name='bert-base-chinese', num_labels=2):
    return BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)

# 加载分类数据
def load_classification_data(data_path):
    texts = []
    labels = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    text, label = parts
                    texts.append(text)
                    labels.append(int(label))
    
    return texts, labels


# 数据处理
def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )

def build_dataset(texts, labels, tokenizer, max_length=128):
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True)
    return dataset

# 训练和验证
def train_classification(data_path, save_weight=True, bert_model_name='bert-base-chinese'):
    # 加载数据
    texts, labels = load_classification_data(data_path)
    num_labels = len(set(labels))
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    # 划分数据
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_dataset = build_dataset(train_texts, train_labels, tokenizer)
    eval_dataset = build_dataset(eval_texts, eval_labels, tokenizer)
    model = build_model(bert_model_name, num_labels)
    training_args = TrainingArguments(
        output_dir='./results',
        run_name=f"bert_sft_{datetime.now():%Y%m%d_%H%M%S}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()}
    )
    trainer.train()
    if save_weight:
        model.save_pretrained('./results')
        tokenizer.save_pretrained('./results')
    return model

# 预测函数
def predict_single_text(model, tokenizer, text, max_length=128):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        prediction = torch.argmax(logits, dim=-1).item()
    return prediction


if __name__ == "__main__":
    data_file = "classification_sample_data.txt"
    model = train_classification(data_file, True)
    # 预测
    model = BertForSequenceClassification.from_pretrained('./results')
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('./results')
    test_file = "classification_test_data.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            title = line.strip()
            if not title:
                continue
            pred = predict_single_text(model, tokenizer, title)
            print(f"{title}\t{pred}")
