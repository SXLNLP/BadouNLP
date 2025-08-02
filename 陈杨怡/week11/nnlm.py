#coding:utf8

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# 数据集构造
class NewsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 模型定义
class BertSFT:
    def __init__(self, model_name, num_labels):
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def train(self, dataset, epochs=3, batch_size=16, lr=5e-5):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# 主程序
if __name__ == "__main__":
    file_path = 'news_data.csv'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = NewsDataset(file_path, tokenizer, max_length=128)

    model = BertSFT('bert-base-chinese', num_labels=5)  # 假设有5个类别
    model.train(dataset, epochs=3, batch_size=16)

    # 保存模型
    model.model.save_pretrained('./sft_model')
    model.tokenizer.save_pretrained('./sft_model')
