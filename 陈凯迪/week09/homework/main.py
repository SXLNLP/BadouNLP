# -*- coding: utf-8 -*-

import logging
import os
import random

import numpy as np
import torch

from config import Config
from evaluate import Evaluator
from loader import load_data
from model import TorchModel, choose_optimizer

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model_config = BertConfig.from_pretrained(config["bert_path"])
    model = TorchModel(model_config, config)
    model = model.from_pretrained(config["bert_path"], config=model_config, model_config=config)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch {epoch} begin")
        train_loss = []

        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            input_ids, attention_mask, labels = batch_data
            loss = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if index % int(len(train_data) / 2) == 0:
                logger.info(f"batch {index}/{len(train_data)} loss: {loss.item():.4f}")

        avg_loss = np.mean(train_loss)
        logger.info(f"epoch {epoch} average loss: {avg_loss:.4f}")
        evaluator.eval(epoch)

        # 保存模型
        model.save_pretrained(os.path.join(config["model_path"], f"epoch_{epoch}"))

    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
