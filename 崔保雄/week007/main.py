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
from datetime import datetime
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
# logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers="StreamHandler")
# logger = logging.getLogger(__name__)


""" 日志记录 """
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_dir = f"logs/{datetime.now().strftime('%Y%m%d')}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
file_handler = logging.FileHandler(log_path, encoding= "utf-8")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式化器
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

"""
模型训练主程序
"""

"""设置CPU生成随机数的种子，以使得结果是确定的，方便下次复现实验结果"""
seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"] + "/" + config["model_type"]):
        os.mkdir(config["model_path"]+ "/" + config["model_type"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # tran_data_list = enumerate(train_data)
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
    acc = 0 #最后一轮训练的准确率
    time_cost = 0 #最后一轮预测的每100条耗时
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("第 %d 轮训练 begin" % epoch)
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
        logger.info("第 %d 轮训练 end" % epoch)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc, time_cost = evaluator.eval(epoch)
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, time_cost

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn", 'bert', 'lstm']:
    for model in ["gated_cnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        acc, time_cost = main(Config)
                        print("最后一轮准确率：", acc, "最后一轮每100条平均耗时：", time_cost, "当前配置：", Config)
                        logger.info(f"训练完成。【{model}，{lr}，{ hidden_size}，{pooling_style}】最后一轮准确率：{acc}，最后一轮每100条平均耗时：{time_cost}，当前配置：{Config}")



