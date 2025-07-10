# -*- coding: utf-8 -*-

import itertools
import torch
import os
import random
import os
import numpy as np
import logging
import csv
from config import Config
from 許家偉.week07.model import TorchModel, choose_optimizer
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
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

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
        acc = main(Config)
        results.append({
            "model": model,
            "learning_rate": lr,
            "batch_size": batch_size,
            "acc": acc,
        })
    # 将结果写入CSV文件
    with open('results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["model", "learning_rate", "batch_size", "acc"])
        writer.writeheader()
        writer.writerows(results)


