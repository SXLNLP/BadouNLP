# evaluate.py
import time
import torch
from sklearn.metrics import accuracy_score
from config_homework import Config
from model import BaseModel
from loader import tokenizer
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_eval(mode, train_loader, val_loader, df):
    model = BaseModel(mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(Config["epoch"]):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 验证
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
    acc = accuracy_score(truths, preds)

    # 推理时间（100条）
    sample_texts = df['text'].tolist()[:100]
    sample_encodings = tokenizer(sample_texts, truncation=True, padding=True, max_length=Config["max_length"], return_tensors='pt')
    sample_ids = sample_encodings['input_ids'].to(device)
    sample_mask = sample_encodings['attention_mask'].to(device)

    start = time.time()
    with torch.no_grad():
        model(sample_ids, sample_mask)
    end = time.time()

    return acc, end - start
