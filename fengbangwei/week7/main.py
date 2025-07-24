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
import pandas as pd
import json
import ast
import time
import copy
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


def main(config, model_index):
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
    evaluator = Evaluator(config, model, logger, sample_count=None)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
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
            # 当索引值为数据集长度一半的整数倍时，执行特定操作
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % model_index)
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc


def predict():
    data = {
        "model": [],
        "learning_rate": [],
        "hidden_size": [],
        "batch_size": [],
        "pooling_style": [],
        "acc": [],
        "100_acc": [],
        "elapsed_time": [],
        "config": []

    }
    df = pd.read_csv('model_eval.csv', encoding='utf-8')
    for model_index, value in enumerate(df.values):
        # 安全地将字符串转为字典 加载配置
        config = ast.literal_eval(value[6])
        # 加载训练数据
        print(config)
        # config["model_type"] = value[0]
        model_index += 1
        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % model_index )

        # 加载模型
        model = TorchModel(config)
        model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重  防止潜在的反序列化攻击
        logger.info("开始预测第%d个模型效果：" % model_index)
        start_time = time.time()  # 开始计时
        model.eval()
        # 加载效果测试类
        evaluator = Evaluator(config, model, logger, sample_count=100)
        acc = evaluator.eval(model_index)
        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time
        print("准确率", acc)
        print(f"预测耗时: {elapsed_time:.4f} 秒")

        data["model"].append(config["model_type"])
        data["learning_rate"].append(config["learning_rate"])
        data["hidden_size"].append(config["hidden_size"])
        data["batch_size"].append(config["batch_size"])
        data["pooling_style"].append(config["pooling_style"])
        data["acc"].append(value[5])
        data["100_acc"].append(acc)
        data["elapsed_time"].append(elapsed_time)
        data["config"].append(config)

    df = pd.DataFrame(data)
    # 写入Excel文件
    df.to_csv("model_eval_result.csv", index=False)


if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    data = {
        "model": [],
        "learning_rate": [],
        "hidden_size": [],
        "batch_size": [],
        "pooling_style": [],
        "acc": [],
        "config": []
    }

    # model_index = 0
    # for model in ["gated_cnn", 'bert', 'lstm']:
    #     Config["model_type"] = model
    #     for lr in [1e-3]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", 'max']:
    #                     Config["pooling_style"] = pooling_style
    #                     model_index += 1
    #                     acc = main(Config, model_index)
    #                     print("最后一轮准确率：", acc, "当前配置：", Config)
    #                     data["model"].append(Config["model_type"])
    #                     data["learning_rate"].append(Config["learning_rate"])
    #                     data["hidden_size"].append(Config["hidden_size"])
    #                     data["batch_size"].append(Config["batch_size"])
    #                     data["pooling_style"].append(Config["pooling_style"])
    #                     data["acc"].append(acc)
    #                     data["config"].append(copy.deepcopy(Config))
    #
    # for index, model in enumerate(['lstm', 'bert']):
    #     Config["model_type"] = model
    #     acc = main(Config, index)
    #     print("最后一轮准确率：", acc, "当前配置：", Config)
    #     data["model"].append(Config["model_type"])
    #     data["learning_rate"].append(Config["learning_rate"])
    #     data["hidden_size"].append(Config["hidden_size"])
    #     data["batch_size"].append(Config["batch_size"])
    #     data["pooling_style"].append(Config["pooling_style"])
    #     data["acc"].append(acc)
    #     data["config"].append(copy.deepcopy(Config))

    # df = pd.DataFrame(data)
    # # 写入csv文件
    # df.to_csv("model_eval.csv", index=False)

    predict()
