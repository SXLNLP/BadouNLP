import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序（三元组损失版本）
"""


def main(config):
    # 创建模型保存目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载数据（三元组格式）
    train_data = load_data(config["train_data_path"], config)

    # 初始化模型
    model = SiameseNetwork(config)

    # 设备配置
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("使用GPU加速训练")
        model = model.cuda()

    # 优化器
    optimizer = choose_optimizer(config, model)

    # 评估器
    evaluator = Evaluator(config, model, logger)

    # 训练循环
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"第 {epoch} 轮训练开始")
        train_loss = []

        for batch in train_data:
            optimizer.zero_grad()
            if cuda_flag:
                batch = [x.cuda() for x in batch]

            # 三元组数据：anchor, positive, negative
            anchor, positive, negative = batch
            loss = model(anchor, positive, negative)

            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # 打印本轮损失
        avg_loss = np.mean(train_loss)
        logger.info(f"第 {epoch} 轮平均损失: {avg_loss:.4f}")

        # 每轮训练后评估
        evaluator.eval(epoch)


        if epoch==10:
            # 保存模型
            model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)

    logger.info("训练完成")


if __name__ == "__main__":
    main(Config)
