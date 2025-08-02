from transformers import get_scheduler
from dataloader import model, NewsData, collote_fn, tokenizer,device
from train import train_loop
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

"""
新闻数据sft
"""
learning_rate = 2e-5
epoch_num = 3
batch_size = 16

train_data = NewsData("train.json")
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

def generate_answer(test_model, title):
    input_seq = f"{title}"
    inputs = tokenizer(
        input_seq,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output = test_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    result = tokenizer.decode(output[0], skip_special_tokens=False).replace(" ", "")
    return result

if __name__ == "__main__":
    total_loss = []
    loss = 0
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        # train
        loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, loss)
        epoch_loss = loss / ((t + 1) * len(train_dataloader))
        total_loss.append(epoch_loss)
        print(f"Epoch {t+1}/train_loss:{epoch_loss}")
        torch.save(model.state_dict(), f'epoch_{t + 1}_model_weights.bin')
        # print(generate_answer(model, "《哪吒2》将于8月2日全网上线"))