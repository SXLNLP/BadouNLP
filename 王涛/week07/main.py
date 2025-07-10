import csv
import itertools

import pandas as pd
import torch
import os
import random
import os
import numpy as np
import logging

from sklearn.model_selection import train_test_split

from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
模型训练主程序
"""


seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    train_data = load_data(config["train_data_path"],config)
    model = TorchModel(config)
    # 优化器
    optimizer = choose_optimizer(config,model)
    # 测试
    evaluator = Evaluator(config,model,logger)
    # 训练
    for epoch in range(config["epoch"]):
        model.train()
        logger.info(f"model: {config['model_type']} epoch {epoch} begin")
        train_loss = []
        for index,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            comment,label = batch_data
            loss = model(x=comment,target=label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            logger.info("batch loss %f"%loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc,infer_time  = evaluator.eval(epoch)
    return acc,infer_time

if __name__ == "__main__":
    param_grid = {
        "batch_size": [100],
        'lr': [1e-3],
        'model': ["gated_cnn", 'bert', 'lstm']
    }
    param_combinations = list(itertools.product(
        param_grid["batch_size"],
        param_grid["lr"],
        param_grid["model"],
    ))
    results = []
    for batch_size,lr,model in param_combinations:
        Config["batch_size"] = batch_size
        Config["learning_rate"] = lr
        Config["model_type"] = model
        acc,infer_time = main(Config)
        results.append({
            "model": model,
            "learning_rate": lr,
            "batch_size": batch_size,
            "acc": acc,
            "time": infer_time
        })
    # 将结果写入CSV文件
    with open('results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["model", "learning_rate", "batch_size", "acc", "time"])
        writer.writeheader()
        writer.writerows(results)
