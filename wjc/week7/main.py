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
import time
import pandas as pd

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
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


def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
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
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc


# 添加计时和结果记录功能
if __name__ == "__main__":
    results = []
    models = []
    lr = []
    hs = []
    bs = []
    ps = []
    oz = []
    if Config["enviroment"] == "produut":
        models = ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn",
                  "stack_gated_cnn", "rcnn", "bert", "bert_lstm",
                  "bert_cnn", "bert_mid_layer"]
        lr = [0.001, 0.0001]
        hs = [32, 64, 128]
        bs = [32, 64, 128]
        ps = ["max", "avg"]
        oz = ["adam", "sgd"]
    elif Config["enviroment"] == "test":
        models = ["fast_text"]
        lr = [0.001]
        hs = [64]
        bs = [64]
        ps = ["max"]
        oz = ["adam"]
    for model in models:
        Config["model_type"] = model
        for l in lr:
            Config["learning_rate"] = l
            for h in hs:
                Config["hidden_size"] = h
                for b in bs:
                    Config["batch_size"] = b
                    for p in ps:
                        Config["pooling_style"] = p
                        for o in oz:
                            Config["optimizer"] = o
                            start_time = time.time()
                            acc = main(Config)
                            end_time = time.time()
                            training_time = end_time - start_time
                            results.append({
                                "model": model,
                                "accuracy": acc,
                                "training_time": training_time,
                                "model_type": Config["model_type"],
                                "learning_rate": Config["learning_rate"],
                                "hidden_size": Config["hidden_size"],
                                "batch_size": Config["batch_size"],
                                "epoch": Config["epoch"]
                            })
    # 输出结果表格
    result_df = pd.DataFrame(results)
    print("\n模型性能对比:")
    print(result_df[["model", "accuracy", "training_time", "model_type", "learning_rate", "hidden_size", "batch_size",
                     "epoch"]])

    # 保存详细结果
    result_df.to_csv("model_comparison.csv", index=False)
