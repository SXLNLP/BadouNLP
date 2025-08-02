# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from torch.utils.data import DataLoader
from loader import TextMatchTripletDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型训练主程序（支持三元组 Triplet 输入）
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据（返回三元组格式）
    train_data = load_data(config["train_data_path"], config)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=TextMatchTripletDataset.collate_fn  # ✅ 使用类中的 collate_fn
    )

    # 加载模型
    model = SiameseNetwork(config)

    # 标识是否使用 GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU 可用，迁移模型至 GPU")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 定义三元组损失函数
    criterion = torch.nn.TripletMarginLoss(margin=config.get("triplet_margin", 1.0), p=2)

    for index, (a_input_ids, p_input_ids, n_input_ids) in enumerate(train_loader):
        logger.info(f"Batch {index} 输入形状: a={a_input_ids.shape}, p={p_input_ids.shape}, n={n_input_ids.shape}")

    # 开始训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("Epoch %d 开始" % epoch)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            # 解包三元组 (anchor, positive, negative)
            a_input_ids, p_input_ids, n_input_ids = batch_data

            if cuda_flag:
                a_input_ids = a_input_ids.cuda()
                p_input_ids = p_input_ids.cuda()
                n_input_ids = n_input_ids.cuda()

            # 前向传播获取嵌入
            a_vec, p_vec, n_vec = model(a_input_ids, p_input_ids, n_input_ids)

            # 计算 Triplet Loss
            loss = criterion(a_vec, p_vec, n_vec)
            train_loss.append(loss.item())

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 打印日志（可选）
            # if index % 10 == 0:
            #     logger.info("Batch %d Loss: %.4f" % (index, loss.item()))

        avg_loss = np.mean(train_loss)
        logger.info("Epoch %d 平均 Loss: %.4f" % (epoch, avg_loss))

        # 每个 epoch 后评估一次
        evaluator.eval(epoch)

    # 保存最终模型
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    logger.info("模型已保存至：%s" % model_path)

    return


if __name__ == "__main__":
    main(Config)
