# coding:utf8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import random

# 数据集构造
class MaskedLanguageModelingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 随机选择一个位置进行mask
        words = text.split()
        if len(words) > 1:
            mask_index = random.randint(0, len(words) - 1)
            original_word = words[mask_index]
            words[mask_index] = '[MASK]'  # 替换为[MASK]
            masked_text = ' '.join(words)
        else:
            masked_text = text
            original_word = ''

        encoding = self.tokenizer.encode_plus(
            masked_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = self.tokenizer.encode(original_word, add_special_tokens=False)
        labels = torch.full((self.max_length,), -100)  # -100 will be ignored in loss
        if original_word:
            labels[0, mask_index] = label[0]  # 只在mask位置设置标签

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

# BERT模型定义
class BertMaskedLM:
    def __init__(self, model_name):
        self.model = BertForMaskedLM.from_pretrained(model_name)
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

    def generate_text(self, prompt, max_length=20):
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(tokens)
                predictions = outputs.logits
                predicted_index = torch.argmax(predictions[0, -1]).item()
                predicted_token = self.tokenizer.decode(predicted_index)

                if predicted_token == '[SEP]':
                    break
                tokens = torch.cat((tokens, torch.tensor([[predicted_index]])), dim=1)

        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

# 主程序
if __name__ == "__main__":
    # 假设你有一个文本列表
    texts = [
        "今天是个好天气",
        "我喜欢学习机器学习",
        "BERT模型非常强大"
    ]
    
    # 初始化数据集和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = MaskedLanguageModelingDataset(texts, tokenizer, max_length=32)

    model = BertMaskedLM('bert-base-chinese')
    model.train(dataset, epochs=3, batch_size=2)

    # 生成文本示例
    generated_text = model.generate_text("今天是个好", max_length=10)
    print("生成的文本:", generated_text)
