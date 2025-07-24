import logging
import os

import numpy as np
import torch
import random
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from config import Config
from nn_pipline2.evaluate import Evaluator
from nn_pipline2.loader import load_data, generate_data
from nn_pipline2.model import TorchModel, choose_optimizer
from tabulate import tabulate
import itertools


# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
torch.manual_seed(seed)
if Config["gpu_switch"]:
    torch.cuda.manual_seed_all(seed)


# 设置混合精度
# scaler = GradScaler()


def main(config):
    acc = 0
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config)
    model = TorchModel(config)

    cuda_flag = config["gpu_switch"]
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.to("mps")
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
            if cuda_flag:
                batch_data = [d.to("mps") for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with autocast():
                loss = model(input_ids, labels)
            loss.backward()
            # 使用梯度缩放防止梯度下溢
            # scaler.scale(loss).backward()
            # 更新参数
            # scaler.step(optimizer)
            # 更新缩放器
            # scaler.update()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch, config)

    model_name = (f"{Config['model_type']}_{Config['learning_rate']}"
                  f"_{Config['hidden_size']}_{Config['batch_size']}"
                  f"_{Config['pooling_style']}_{acc}")
    torch.save(model.state_dict(), os.path.join(config["model_path"], f"{model_name}.pth"))

    return acc


if __name__ == '__main__':
    # main(Config)

    # for model in ["gated_cnn", 'bert', 'lstm']:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", 'max']:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)

    results = []

    # 定义参数组合
    models = ["gated_cnn", "lstm"]
    learning_rates = [1e-3, 1e-4]
    hidden_sizes = [128]
    batch_sizes = [64, 128]
    pooling_styles = ["avg", "max"]

    # 遍历参数组合
    for model, lr, hidden_size, batch_size, pooling_style in itertools.product(
            models, learning_rates, hidden_sizes, batch_sizes, pooling_styles
    ):
        # 更新配置
        Config["model_type"] = model
        Config["learning_rate"] = lr
        Config["hidden_size"] = hidden_size。
        Config["batch_size"] = batch_size
        Config["pooling_style"] = pooling_style

        # 运行训练并获取准确率
        accuracy = main(Config)

        # 存储结果
        results.append({
            "模型类型": model,
            "学习率": lr,
            "隐藏层大小": hidden_size,
            "批次大小": batch_size,
            "池化方式": pooling_style,
            "最后一轮准确率": accuracy
        })

    # 输出表格
    print(tabulate(results, headers="keys", tablefmt="grid"))
    with open("experiment_results.txt", "w", encoding="utf-8") as f:
        f.write(tabulate(results, headers="keys", tablefmt="grid"))
