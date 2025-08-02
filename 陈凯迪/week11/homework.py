import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import random

# 设置随机种子以确保可重现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 加载和预处理数据
def load_news_dataset(sample_size=5000):
    """加载新闻数据集并进行预处理"""
    try:
        # 尝试从Hugging Face加载数据集
        dataset = load_dataset("ag_news", split=f"train[:{sample_size}]")
        print("从Hugging Face加载AG News数据集")

        # 将数据集转换为pandas DataFrame
        df = pd.DataFrame(dataset)
        df = df.rename(columns={"text": "content"})
        df["title"] = "Untitled"  # 添加一个虚拟标题列

        # 选择我们需要的列
        df = df[["title", "content"]]

        # 创建一个新的列，将标题和内容组合成模型输入
        df["prompt"] = "新闻标题: " + df["title"] + "\n新闻内容: " + df["content"]

        return df
    except Exception as e:
        print(f"加载Hugging Face数据集失败: {e}")
        print("创建示例数据集...")

        # 创建示例数据集
        data = {
            "title": ["科技突破", "经济动态", "体育新闻"],
            "content": [
                "研究人员开发出新型AI模型，性能提升40%。该模型在多个基准测试中表现优异。",
                "全球股市今日普遍上涨，科技股领涨。分析师认为市场情绪正在改善。",
                "在昨晚的比赛中，主场球队以3-2险胜对手，球迷欢呼雀跃。"
            ]
        }
        df = pd.DataFrame(data)
        df["prompt"] = "新闻标题: " + df["title"] + "\n新闻内容: " + df["content"]
        return df


# 加载数据集
news_df = load_news_dataset()
print(f"数据集大小: {len(news_df)}")
print("示例数据:")
print(news_df["prompt"][0])
print("\n" + "-" * 80 + "\n")


# 2. 创建自定义数据集
class NewsDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]

        # 对文本进行标记化
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 对于SFT，输入和标签是相同的
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# 3. 初始化模型和分词器
MODEL_NAME = "gpt2"  # 使用小模型便于演示
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
print(f"加载模型: {MODEL_NAME}")
print(f"模型参数数量: {model.num_parameters():,}")

# 4. 准备数据集和数据加载器
dataset = NewsDataset(news_df["prompt"].tolist(), tokenizer)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(val_dataset)}")
print(f"批次大小: {BATCH_SIZE}")
print(f"训练批次数: {len(train_loader)}")
print("\n" + "-" * 80 + "\n")

# 5. 设置训练参数
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100

# 优化器和调度器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)


# 6. 训练函数
def train(model, dataloader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Training]", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# 7. 验证函数
def validate(model, dataloader, epoch):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Validation]", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            progress_bar.set_postfix({"val_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# 8. 训练循环
print("开始训练...")
history = {"train_loss": [], "val_loss": []}

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, scheduler, epoch)
    val_loss = validate(model, val_loader, epoch)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
    print("-" * 50)

# 9. 保存微调后的模型
SAVE_PATH = "fine_tuned_news_model"
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n模型已保存到 {SAVE_PATH}")


# 10. 生成示例
def generate_news(model, tokenizer, prompt, max_length=100):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# 测试生成
print("\n生成示例:")
test_prompt = "新闻标题: 科技新突破\n新闻内容:"
generated_news = generate_news(model, tokenizer, test_prompt)
print(generated_news)

# 绘制训练损失图表
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="训练损失")
plt.plot(history["val_loss"], label="验证损失")
plt.title("训练和验证损失")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
