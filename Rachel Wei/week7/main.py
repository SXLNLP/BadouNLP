# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
import time
from datetime import datetime
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import csv
from split_data import split_data

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
log_level = getattr(logging, Config["log_level"].upper(), logging.INFO)
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
    ## Create output folder to save model parameters
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    print("=================== start model training ===================")
    
    ## load training datasets 
    time_start = time.perf_counter()
    train_data = load_data(config["train_data_path"], config)
    time_after_load = time.perf_counter()
    print(f"Loading data takes : {time_after_load - time_start:.4f} seconds")
    
    ## load the model that we need
    model = TorchModel(config)
    print("current model type is ", config['model_type'])
    
    # check if we can use CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 CUDA 加速:", torch.cuda.get_device_name(0))
    elif False and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple M1/M2 的 MPS 加速")
    else:
        device = torch.device("cpu")
        print("使用 CPU 运行")
        model.to(device)

    ## load optimizer
    optimizer = choose_optimizer(config, model)
    
    # #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    ## training process
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
        #     # if cuda_flag:
        #     #     batch_data = [d.cuda() for d in batch_data]
            batch_data = [d.to(device) for d in batch_data]
            optimizer.zero_grad()
            sensitivty_score, review_content = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况. 这里是将 batch_data 拆解为两个部分
            loss = model(review_content, sensitivty_score)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    time_after_training = time.perf_counter()
    timing_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    result_row = {
        "timing_now": timing_now,
        "model_type": Config["model_type"],
        "epoch":Config["epoch"],
        "learning_rate": Config["learning_rate"],
        "hidden_size": Config["hidden_size"],
        "batch_size": Config["batch_size"],
        "pooling_style": Config["pooling_style"],
        "dropout": Config["dropout"],
        "time_spent": f"{time_after_training - time_start:.4f}",
        "accuracy": acc
    }
    log_to_csv("training_results.csv", result_row.keys(), result_row)
    return acc


def log_to_csv(file_path, headers, row):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



if __name__ == "__main__":
    split_data(Config)
    # main(Config)

    # for m in ["cnn"]:
    #     Config["model_type"] = m
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    
    # for m in ["gated_cnn", 'bert', 'lstm']:
    for m in [
              "fast_text"
              , "lstm"
              , "gru"
              , "rnn"
              , "cnn", "gated_cnn", "stack_gated_cnn", "rcnn"
              , "bert", "bert_lstm", "bert_cnn", "bert_mid_layer"
              ]:
    #     Config["model_type"] = m
        # print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    
        Config["model_type"] = m
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 64]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        for dropout in [0.3, 0.5]:
                            Config["dropout"] = dropout
                            print("最后一轮准确率：", main(Config), "当前配置：", Config)

