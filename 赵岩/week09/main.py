# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from model import TorchModel
from loader import load_data
from evaluate import Evaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TorchModel(config)
    model.to(device)

    optimizer = choose_optimizer(config, model)

    # 加载训练数据
    train_loader = load_data(config["train_data_path"], config, shuffle=True)

    evaluator = Evaluator(config, model, logger)

    # 训练循环
    for epoch in range(1, config["epochs"] + 1):
        print(f"epoch {epoch} begin")
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids, attention_masks, labels = [item.to(device) for item in batch]
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
        print(f"epoch {epoch} end, average loss: {total_loss / len(train_loader)}")

        # 评估模型
        evaluator.eval(epoch)

    return model


if __name__ == "__main__":
    from config import Config

    model = main(Config)