import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('csv', data_files={'train': 'news_train.csv', 'test': 'news_test.csv'})

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# 应用预处理函数
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 创建DataLoader
train_dataloader = DataLoader(encoded_dataset['train'], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(encoded_dataset['test'], batch_size=8)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练循环
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    # 验证阶段
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            _, preds = torch.max(outputs.logits, dim=-1)
            correct_predictions += (preds == batch['labels']).sum().item()
            total_predictions += len(preds)

    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{num_epochs} - Accuracy: {accuracy:.4f}")

print("Training complete.")

