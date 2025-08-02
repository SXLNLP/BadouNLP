import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from data_loader import TitleContentDataset
from model import Title2ContentModel
from utils import init_tokenizer, save_model
from config import config


def train():
    # 初始化
    tokenizer = init_tokenizer()
    dataset = TitleContentDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = Title2ContentModel()
    model.resize_token_embeddings(len(tokenizer))  # 适配新增的特殊token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * config.num_epochs
    )

    # 训练循环
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                loss_mask=batch['loss_mask']
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % config.logging_steps == 0:
                print(f"Epoch {epoch + 1} | Step {step + 1} | Loss: {loss.item():.4f}")

            if (step + 1) % config.save_steps == 0:
                save_model(model, tokenizer, config.output_dir)

        print(f"Epoch {epoch + 1} | Avg Loss: {total_loss / len(dataloader):.4f}")

    save_model(model, tokenizer, config.output_dir)


if __name__ == "__main__":
    train()