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
from plottable import Table, ColumnDefinition
import matplotlib.pyplot as plt

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        time = evaluator.get_time()
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, time

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    output_file = "output/result.txt"
    f = open(output_file, "w")
    for model in ["gated_cnn", 'rnn', 'lstm', 'gru']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        rate, time = main(Config)
                        f.write(str(rate) + "--" + str(time) + "--" + str(Config) + "\n")
                        print("最后一轮准确率：", rate, "当前配置：", Config)
    f.close()


    # 绘制表格
    columns = ["model_type", "max_length", "hidden_size", "kernel_size", "num_layers", "batch_size", "epoch", "pooling_style", "optimizer", "learning_rate", "seed", "class_num", "vocab_size", "accuracy_rate", "time(ms/1000colums)"]

    data = []

    with open(output_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("--")
            rate = line[0]
            time = line[1]
            info = eval(line[2])
            info = ([info["model_type"], info["max_length"], info["hidden_size"], info["kernel_size"], info["num_layers"], info["batch_size"], info["epoch"], info["pooling_style"], info["optimizer"], info["learning_rate"], info["seed"], info["class_num"], info["vocab_size"], rate, float(time) * 1000])
            data.append(info)

    table = pd.DataFrame(data, columns=columns)
    fig, ax = plt.subplots()
    Table(table, 
        odd_row_color='#f5f5f5',
        col_label_cell_kw={
            "facecolor": '#a5d8ff'
            }, 
        cell_kw={
            "facecolor": '#e7f5ff'
            },
        textprops={
            "fontsize": 15,
            'fontname': 'Times New Roman',
            'ha' : 'center'
        },
        column_definitions = [
            ColumnDefinition(name='model_type', width=1.5),
            ColumnDefinition(name='max_length', width=1.5),
            ColumnDefinition(name='hidden_size', width=1.5),
            ColumnDefinition(name='kernel_size', width=1.5),
            ColumnDefinition(name='num_layers', width=1.5),
            ColumnDefinition(name='batch_size', width=1.2),
            ColumnDefinition(name='epoch', width=1.2),
            ColumnDefinition(name='pooling_style', width=1.5),
            ColumnDefinition(name='optimizer', width=1.5),
            ColumnDefinition(name='learning_rate', width=1.5),
            ColumnDefinition(name='vocab_size', width=1.5),
            ColumnDefinition(name='accuracy_rate', width=1.5),
            ColumnDefinition(name='time(ms/1000colums)', width=2.8)
        ]
    )
    plt.show()

