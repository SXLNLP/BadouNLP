# -*- coding: utf-8 -*-

import os
import time
import logging
import torch
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(config):
    # 创建模型保存目录
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device
    logger.info(f"使用设备: {device}")

    # 加载数据
    train_data = load_data(config["train_data_path"], config)
    valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    # 初始化模型
    model = TorchModel(config).to(device)

    # 初始化优化器
    optimizer = choose_optimizer(config, model)

    # 初始化评估器
    evaluator = Evaluator(config, model, logger)

    # 训练循环
    best_f1 = 0
    no_improve = 0
    for epoch in range(config["epoch"]):
        logger.info(f"Epoch {epoch + 1}/{config['epoch']}")
        model.train()
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(train_data):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device)
            }

            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0 and step != 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"Step {step}, Loss: {avg_loss:.4f}")

        avg_loss = total_loss / len(train_data)
        logger.info(f"Epoch {epoch + 1} finished, Avg Loss: {avg_loss:.4f}")

        # 评估
        f1 = evaluator.eval(epoch + 1)

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(config["model_path"], "best_model.bin"))
            logger.info(f"New best model saved with F1: {best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= 3:  # 提前停止
                logger.info(f"No improvement for {no_improve} epochs, early stopping...")
                break

    logger.info("Training finished")


if __name__ == "__main__":
    train(Config)
