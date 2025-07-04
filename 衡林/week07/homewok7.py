import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.nn as nn

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
df = pd.read_csv("../文本分类练习.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 创建数据集类
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
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 参数设置
MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

# 初始化模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model = model.to(device)

# 创建数据加载器
def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = ReviewDataset(
        texts=df.review.values,
        labels=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train_model(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(data_loader)

# 评估函数
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, predictions = torch.max(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# 训练循环
for epoch in range(EPOCHS):
    train_loss = train_model(model, train_loader, optimizer)
    val_acc = evaluate_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'bert_classifier.pth')

# 预测函数
def predict_sentiment(review):
    model.eval()
    encoding = tokenizer.encode_plus(
        review,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return "正面评价" if prediction == 1 else "负面评价"

# 测试示例
sample_reviews = [
    "速度快，口感不错，味道足，量大",
    "质量很差，描述不符",
    "物流速度很快，包装完好",
    "客服态度恶劣，解决问题效率低"
]

for review in sample_reviews:
    result = predict_sentiment(review)
    print(f"评论: '{review}' → {result}")
