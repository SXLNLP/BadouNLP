import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
import numpy as np
import pandas as pd


# 配置
class Config:
    model_name = "gpt2-medium"
    dataset_name = "cnn_dailymail"
    max_length = 512
    batch_size = 4
    learning_rate = 2e-5
    epochs = 3
    save_path = "./sft_model"


# 数据预处理
class NewsSFTDataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.dataset = load_dataset(
            Config.dataset_name,
            "3.0.0",
            split=split
        )
        self.tokenizer = tokenizer
        self.max_length = Config.max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        text = sample["article"]
        summary = sample["highlights"]
        prompt = f"{text}\n\n摘要: "

        full_text = prompt + summary + self.tokenizer.eos_token

        # Tokenization
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()

        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )
        labels[:len(prompt_tokens)] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }


# 训练函数
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(Config.model_name)
    model.to(device)

    train_dataset = NewsSFTDataset(tokenizer, split="train[:10%]")
    val_dataset = NewsSFTDataset(tokenizer, split="validation[:5%]")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size
    )

    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.epochs):
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{Config.epochs} | Train Loss: {train_loss:.4f}")

    model.save_pretrained(Config.save_path)
    tokenizer.save_pretrained(Config.save_path)
    print(f"Model saved to {Config.save_path}")

    test_sample = val_dataset[0]
    input_ids = test_sample["input_ids"].unsqueeze(0).to(device)

    outputs = model.generate(
        input_ids,
        max_length=Config.max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Summary:")
    print(generated_text.split("摘要: ")[-1])


if __name__ == "__main__":
    main()
