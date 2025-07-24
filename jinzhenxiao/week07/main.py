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

import csv
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


import time


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
        
    model_path = os.path.join(config["model_path"], "epoch_%s_%d.pth" % (config["model_type"], epoch,))
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, model

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    results = []
    for model_type in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model_type
        acc, model = main(Config)

        # 预测
        start_time = time.time()
        for i in range(100):
            model.eval()
            test_data = load_data(Config["data_path"], Config, shuffle=False)
            with torch.no_grad():
                for index, batch_data in enumerate(test_data):
                    if torch.cuda.is_available():
                        batch_data = [d.cuda() for d in batch_data]
                    input_ids, labels = batch_data
                    model(input_ids)
        end_time = time.time()
        prediction_time = end_time - start_time

        results.append([Config["learning_rate"], Config["model_type"], Config["hidden_size"], acc, prediction_time])
        print(f"learning_rate: {Config['learning_rate']}, model_type: {Config['model_type']}, hidden_size: {Config['hidden_size']}, acc: {acc}, prediction_time: {prediction_time}")
    
    # 结果写入 csv
    with open("output/results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Learning_rate", "Model", "hidden_size", "acc", "time(预测100耗时)"])
        writer.writerows(results)


