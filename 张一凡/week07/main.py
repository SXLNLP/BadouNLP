# -*- coding: utf-8 -*-
import os
import time
import torch
import random
import numpy as np
import pandas as pd
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_speed(model, data_loader, num_samples=100):
    model.eval()
    start_time = time.time()
    count = 0

    with torch.no_grad():
        for inputs, _ in data_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            _ = model(inputs)
            count += inputs.size(0)

            if count >= num_samples:
                break

    return time.time() - start_time


def train(config):
    set_seed(config["seed"])

    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])

    # 准备数据
    train_loader = load_data(config["train_data_path"], config, shuffle=True)
    valid_loader = load_data(config["valid_data_path"], config, shuffle=False)

    # 初始化模型
    model = TorchModel(config)
    if torch.cuda.is_available():
        model = model.cuda()

    # 优化器
    optimizer = choose_optimizer(config, model)

    # 评估器
    evaluator = Evaluator(config, model, logger)

    best_acc = 0
    for epoch in range(config["epoch"]):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch + 1}/{config['epoch']}], "
                            f"Step [{batch_idx + 1}/{len(train_loader)}], "
                            f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{config['epoch']}], "
                    f"Average Loss: {avg_loss:.4f}")

        # 评估
        acc = evaluator.eval(epoch + 1)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            model_path = os.path.join(
                config["model_path"],
                f"{config['model_type']}_best.pth"
            )
            torch.save(model.state_dict(), model_path)
            logger.info(f"模型保存到 {model_path}")

    # 测试预测速度
    test_speed = evaluate_speed(model, valid_loader)
    logger.info(f"预测100条评论耗时: {test_speed:.2f}秒")

    return best_acc, test_speed


def run_experiments():
    results = []
    base_config = Config.copy()

    for model_type in ["bert", "lstm", "cnn"]:
        for lr in [1e-3, 1e-4]:
            for hidden_size in [128, 256]:
                config = base_config.copy()
                config.update({
                    "model_type": model_type,
                    "learning_rate": lr,
                    "hidden_size": hidden_size
                })

                logger.info(f"\n开始实验: {config}")
                try:
                    acc, speed = train(config)
                    results.append({
                        "Model": model_type,
                        "Learning_Rate": lr,
                        "Hidden_Size": hidden_size,
                        "Accuracy": round(acc, 4),
                        "Time_per_100(s)": round(speed, 2)
                    })
                except Exception as e:
                    logger.error(f"实验失败: {str(e)}")

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv("experiment_results.csv", index=False)
    logger.info("\n实验结果:\n" + str(result_df))


if __name__ == "__main__":
    run_experiments()
