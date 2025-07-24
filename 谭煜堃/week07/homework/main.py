# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluator import Evaluator
from loader import GlobalDataManager
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

model_type = Config["model_type"]
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
epoch = Config["epoch"]


def main(config):

    #加载数据
    gdm = GlobalDataManager(config)
    #加载模型
    model = TorchModel(config)
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    acc_hist = None
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        #print(f"gdm.train_data: {gdm.train_data}")
        for index, batch_data in enumerate(gdm.train_data):
            optimizer.zero_grad()
            input_ids, labels = batch_data
            # print(f"batch_data: {batch_data}")
            # print(f"input_ids.shape: {input_ids.shape}")
            # print(f"labels.shape: {labels.shape}")
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % 20 == 0:
                logger.info("epoch %d, batch %d, loss %f" % (epoch, index, np.mean(train_loss)))
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        logger.info("epoch %d, accuracy: %f" % (epoch, acc))
        if acc_hist is not None:
            if acc < acc_hist:
                logger.info("当前准确度 %f 小于历史准确的 %f,按早停策略放弃继续训练" % (acc,acc_hist))
                break #早停
        acc_hist = acc
    
    # 正确创建目录路径（不含文件名）
    model_dir = os.path.join(config["model_path"], "model-%s" % model_type)  # 注意这里去掉了文件名
    # 确保目录存在（递归创建）
    os.makedirs(model_dir, exist_ok=True)  # 使用 exist_ok 避免重复创建错误

    # 创建完整的文件路径
    model_path = os.path.join(model_dir, "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    
    return acc

if __name__ == "__main__":
    main(Config)

