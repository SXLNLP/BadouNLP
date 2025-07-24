# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序 - 适配BERT
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
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
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            if cuda_flag:
                # 将数据转移到GPU
                batch_data = {
                    "input_ids": batch_data["input_ids"].cuda(),
                    "attention_mask": batch_data["attention_mask"].cuda(),
                    "labels": batch_data["labels"].cuda()
                }

            # 获取输入数据
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]

            # 计算损失
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss.item())

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

    # 保存模型
    model_path = os.path.join(config["model_path"], "bert_ner_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    logger.info("模型保存完成: %s", model_path)

    return model, train_data


if __name__ == "__main__":
    # 添加BERT相关配置
    Config["bert_path"] = "bert-base-chinese"
    # 标签数量需要根据schema文件确定
    # Config["class_num"] = len(DataGenerator.load_schema(Config["schema_path"]))

    model, train_data = main(Config)
