# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import time
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
    import time
    from data_process import buildData
    import os
    import csv

    # 生成训练数据
    buildData()

    start_time = time.time()
    # 开始调用模型
    accuracy = main(Config)
    end_time = time.time()

    model_type = Config["model_type"]
    hidden_size = Config["hidden_size"]
    kernel_size = Config["kernel_size"]
    num_layers = Config["num_layers"]
    batch_size = Config["batch_size"]
    learning_rate = Config["learning_rate"]
    training_time = end_time - start_time
    time_consumption = round(training_time, 2)

    csv_file_path = r"..\week7_text_csv\week7_assignment.csv"  # 数据存储的目标文件
    new_data = [
        model_type, learning_rate, hidden_size,
        batch_size, kernel_size, num_layers,
        accuracy, time_consumption
    ],
    if not os.path.isfile(csv_file_path):
        # 初始数据
        initial_data = [
            [
                "model_type", "learning_rate", "hidden_size",
                "batch_size", "kernel_size", "num_layers",
                "accurate", "time_consumption"],  # 标题行
            new_data,
        ]

        # 新建文件并写入初始数据（mode='w'）
        with open("data.csv", mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(initial_data)  # 写入所有行
    else:
        # 如果文件已存在，则追加数据（mode='a'）
        with open("data.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(new_data)  # 写入新数据


    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg"]:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)
