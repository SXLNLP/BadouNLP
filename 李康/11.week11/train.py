import torch
from tqdm.auto import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
训练代码
"""
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, data in enumerate(dataloader, start=1):
        optimizer.zero_grad()

        data = data.to(device)
        pred = model(**data)

        loss = pred.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss