# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import time
import logging
import json
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from torch.utils.data import DataLoader

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"
handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
logger.addHandler(handler)

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
    
    #加载模型
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    muti_gpu_flag = False
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        device_ids = list(range(config["num_gpus"]))
        if len(device_ids) > 1:
            logger.info("使用多卡gpu训练")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            muti_gpu_flag = True
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_dataset = load_data(config["train_data_path"], config, logger)
    train_data = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch in enumerate(train_data):
            if cuda_flag:
                batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if muti_gpu_flag:
                loss = torch.mean(loss)
            train_loss.append(float(loss))
            if index % max(1, int(len(train_data) / 2)) == 0 and index != 0:
                logger.info("batch loss %f" % loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":

    main(Config)


