import torch
from transformers import BertForMaskedLM
from loader import load_data
from config import Config

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载BERT自回归mask模型
model = BertForMaskedLM.from_pretrained(Config["bert_model_path"])
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.get("lr", 1e-5))

# 加载数据
train_loader = load_data(Config["train_data_path"], Config, logger=None, shuffle=True)

for epoch in range(Config.get("epochs", 3)):
    model.train()
    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}") 