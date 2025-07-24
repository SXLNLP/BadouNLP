import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# 1. 数据加载与预处理
def load_data(file_path):
    texts, labels = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 手动分割（防止特殊逗号问题）
            parts = line.split(',', 1)  # 只分割第一个逗号
            
            # 验证格式
            if len(parts) != 2:
                continue
                
            label, text = parts
            label = label.strip()
            
            # 验证标签有效性
            if label not in {'0', '1'}:
                continue
                
            labels.append(int(label))
            texts.append(text)
    
    return texts, labels

# 2. 数据集构建
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# 3. 主程序
if __name__ == "__main__":
    print("参数设置")
    DATA_PATH = 'F:/BaiduNetdiskDownload/八斗精品班/第七周 文本分类/week7 文本分类问题/文本分类练习.csv'
    MAX_LENGTH = 128
    BATCH_SIZE = 2
    
    
    print("加载数据")
    texts, labels = load_data(DATA_PATH)
    
    print("划分数据集")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print("初始化分词器")
    tokenizer = BertTokenizer.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)
    
    print("文本编码")
    train_encodings = tokenizer(
        train_texts, truncation=True, padding='max_length', 
        max_length=MAX_LENGTH, return_tensors='pt'
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding='max_length', 
        max_length=MAX_LENGTH, return_tensors='pt'
    )
    
    print("创建数据集")
    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)
    
    print("加载预训练模型")
    model = BertForSequenceClassification.from_pretrained(
        r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", num_labels=2
    )
    
    print("训练参数配置")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )
    
    print("创建Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    print("开始训练...")
    trainer.train()
    
    print("进行评估...")
    predictions = trainer.predict(val_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)
    
    print("\n分类报告:")
    print(classification_report(val_labels, preds, target_names=['差评(0)', '好评(1)']))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(val_labels, preds))
