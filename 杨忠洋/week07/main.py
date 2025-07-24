# -*- coding: utf-8 -*-
import logging
import os
import random

import numpy as np
import pandas as pd
import torch

from config import Config
from evaluate import Evaluator
from loader import load_data
from model import TorchModel, choose_optimizer

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 结果表格
results = []


def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
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
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            # acc = evaluator.eval(epoch)
            # scheduler.step(acc)

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    # 训练完成后测试速度
    speed = evaluator.test_speed()

    # 收集结果
    result = {
        "Model": config["model_type"],
        "Learning_Rate": config["learning_rate"],
        "Hidden_Size": config["hidden_size"],
        "Batch_Size": config["batch_size"],
        "Pooling_Style": config["pooling_style"],
        "Accuracy": acc,
        "Inference_Time(ms)": speed
    }
    results.append(result)

    return acc


if __name__ == "__main__":
    # 运行不同模型
    models = ["bert", "lstm", "gru"]

    for model in models:
        Config["model_type"] = model

        # BERT需要更小的学习率
        if model == "bert":
            Config["learning_rate"] = 2e-5
        else:
            Config["learning_rate"] = 1e-3

        print(f"\n{'=' * 30}")
        print(f"训练 {model.upper()} 模型")
        print(f"{'=' * 30}")

        main(Config)

    # 输出结果表格
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("\n模型性能对比:")
    print(df)
