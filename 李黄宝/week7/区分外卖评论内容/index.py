# 环境安装（取消注释运行）
# !pip install transformers torch pandas datasets scikit-learn accelerate

import torch
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import accelerate
from accelerate import Accelerator

# 初始化加速器（自动处理设备）
accelerator = Accelerator()
device = accelerator.device
print(f"使用设备: {device}")

# 1. 数据准备 - 适配您的CSV格式
df = pd.read_csv("../文本分类练习.csv")

# 检查数据
print(f"数据集大小: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print(f"标签分布:\n{df['label'].value_counts()}")

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. 创建PyTorch数据集
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. 初始化模型和分词器
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 64  # 减少序列长度以节省内存
BATCH_SIZE = 8  # 减少批大小以适应苹果电脑内存
EPOCHS = 2  # 减少训练轮次
LEARNING_RATE = 2e-5

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 创建数据加载器
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        texts=df.review.values,
        labels=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# 使用accelerate准备模型和优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 使用accelerate准备模型和数据加载器
model, optimizer, train_data_loader, scheduler = accelerator.prepare(
    model, optimizer, train_data_loader, scheduler
)

# 5. 训练函数 - 使用accelerate进行优化
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    total_batches = len(data_loader)
    start_time = time.time()
    
    for batch_idx, batch in enumerate(data_loader):
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 每10个批次打印一次进度
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / (batch_idx + 1)
            remaining_batches = total_batches - (batch_idx + 1)
            remaining_time = remaining_batches * avg_time_per_batch
            print(f"批次 {batch_idx+1}/{total_batches} - 损失: {loss.item():.4f} - 预计剩余时间: {remaining_time/60:.1f}分钟")
    
    return total_loss / total_batches

# 6. 评估函数
def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            # 使用accelerator收集所有设备的结果
            predictions.extend(accelerator.gather(preds).cpu().numpy())
            actual_labels.extend(accelerator.gather(batch['labels']).cpu().numpy())
    
    accuracy = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions, average='weighted')
    return accuracy, f1

# 7. 训练循环
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    epoch_start = time.time()
    train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler)
    epoch_time = time.time() - epoch_start
    
    print(f'训练损失: {train_loss:.4f} - 耗时: {epoch_time/60:.1f}分钟')
    
    # 评估模型
    val_acc, val_f1 = eval_model(model, test_data_loader, device)
    print(f'验证准确率: {val_acc:.4f}, F1分数: {val_f1:.4f}\n')

# 8. 保存模型
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("bert_ecommerce_classifier")
tokenizer.save_pretrained("bert_ecommerce_classifier")
print("模型已保存")

# 9. 测试推理
def predict_sentiment(review, model, tokenizer, device, max_len=64):
    model.eval()
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
    
    return "正面评价" if prediction.item() == 1 else "负面评价"

# 加载保存的模型进行推理
loaded_model = BertForSequenceClassification.from_pretrained("bert_ecommerce_classifier")
loaded_model = loaded_model.to(device)
loaded_model.eval()

# 示例预测
sample_reviews = [
    "很快，好吃，味道足，量大",  # 您的示例
    "质量很差，完全不值这个价",
    "物流速度很快，包装完好",
    "客服态度恶劣，解决问题效率低"
]

print("\n测试预测:")
for review in sample_reviews:
    print(f"评论: '{review}'")
    print(f"预测: {predict_sentiment(review, loaded_model, tokenizer, device)}\n")
