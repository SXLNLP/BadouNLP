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
import csv

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
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
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data, valid_data = load_data(config["data_path"], config)
    config['train_data_size'] = len(train_data.dataset)
    config['valid_data_size'] = len(valid_data.dataset)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger, valid_data)
    train_time = 0
    valid_time = 0
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        start_time = time.time()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        train_time += time.time() - start_time
        start_time = time.time()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        valid_time += time.time() - start_time
        model_path = os.path.join(config["model_path"], "%s_model_%d.pth" % (config['model_type'], config["epoch"]))
        torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc, train_time, valid_time


if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    headers = ["Model", "Learning_Rate", "Hidden_Size", "Epoch", "Batch_Size", "Train_Data_Size", "Valid_Data_Size",
               "Vacob_Size", "Acc", "Train_Time(秒/1轮)", "Valid_Time(秒/1轮)", "Predict_Time(秒/100条)"]
    csv_data = []
    for model in ["rnn", "lstm", 'cnn']:
        Config["model_type"] = model
        acc, train_time, valid_time = main(Config)
        row = []
        row.append(model)
        row.append(Config['learning_rate'])
        row.append(Config['hidden_size'])
        row.append(Config['epoch'])
        row.append(Config['batch_size'])
        row.append(Config['train_data_size'])
        row.append(Config['valid_data_size'])
        row.append(Config['vocab_size'] + 1)
        row.append(acc)
        row.append(train_time / Config['epoch'])
        row.append(valid_time / Config['epoch'])
        row.append(valid_time / Config['epoch'] / Config['valid_data_size'] * 100)
        csv_data.append(row)
    # 写入CSV文件
    with open("output.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 写入表头（可选）
        writer.writerow(headers)
        # 写入多行数据
        writer.writerows(csv_data)
