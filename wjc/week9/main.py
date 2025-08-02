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
from torchcrf import CRF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, shuffle=True)

    # 加载模型
    model = TorchModel(config)

    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可用，迁移模型至GPU")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epoch"] * len(train_data),
        eta_min=1e-6
    )

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练循环
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch {epoch} 开始")
        train_loss = []
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"当前学习率: {current_lr:.2e}")

        # 在第10轮后启用CRF
        if epoch == 10 and config["use_crf"]:
            logger.info("启用CRF层")
            model.use_crf = True
            model.crf_layer = CRF(config["class_num"], batch_first=True)
            if cuda_flag:
                model.crf_layer = model.crf_layer.cuda()

        # 在第5轮后解冻BERT
        if epoch == 5 and config.get("freeze_bert", False):
            logger.info("解冻BERT参数")
            for param in model.bert.parameters():
                param.requires_grad = True

        for batch_index, batch in enumerate(train_data):
            optimizer.zero_grad()

            # 准备数据
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            if cuda_flag:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            # 前向传播和损失计算
            loss = model(input_ids, attention_mask, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if batch_index % 10 == 0:
                logger.info(f"batch {batch_index}/{len(train_data)} loss: {loss.item():.4f}")

        # 计算平均损失
        avg_loss = np.mean(train_loss)
        logger.info(f"epoch {epoch} 平均损失: {avg_loss:.4f}")

        # 评估模型
        evaluator.eval(epoch)

        # 保存模型
        model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型已保存至 {model_path}")

    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)
