# -*- coding: utf-8 -*-
import time

import pandas as pd
import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from performance import Performance
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from dataProcess import DataProcessor

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
    acc = 0
    time_cost = 0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        start = time.perf_counter()
        acc += evaluator.eval(epoch)
        end = time.perf_counter()
        time_cost += end - start

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    # 预测平均耗时
    acc /= config["epoch"]
    time_cost /= config["epoch"]
    Performance["acc"].append(round(acc, 4))
    Performance["time(s)"].append(round(time_cost, 4))
    return acc    # 平均准确率


# 将Config中的值赋值给Performance
def config2Performance(config, performance):
    performance["model_type"].append(config["model_type"])
    performance["learning_rate"].append(config["learning_rate"])
    performance["batch_size"].append(config["batch_size"])
    performance["hidden_size"].append(config["hidden_size"])


# 将Performance的值整理成表格输出
def output_performance_table(performances):
    # 创建DataFrame
    df = pd.DataFrame(performances)

    # 导出到Excel
    df.to_excel('./output/performance.xlsx', index=False, sheet_name='sheet1')
    print("Excel文件已创建: performance.xlsx")


if __name__ == "__main__":
    # 处理初始样本数据
    DataProcessor(Config["input_path"], Config["text_column"], Config["label_column"], Config["proportion"])
    # main(Config)
    # config2Performance(Config, Performance)
    # output_performance_table(Performance)
    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ['max']:
                        Config["pooling_style"] = pooling_style
                        main(Config)
                        config2Performance(Config, Performance)
                        # print("最后一轮准确率：", main(Config), "当前配置：", Config)
    output_performance_table(Performance)
