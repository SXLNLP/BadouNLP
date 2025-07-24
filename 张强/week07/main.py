# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import time
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from tabulate import tabulate
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""
results = []

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
            # input_ids, attention_mask, labels = batch_data
            loss = model(input_ids, labels)
            # loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        # 评估并记录时间
        start_time = time.time()
        acc = evaluator.eval(epoch)
        eval_time = time.time() - start_time

        # 如果是最后一轮，记录结果
        if epoch == config["epoch"]:
            return acc, eval_time

    return acc, 0  # 如果未执行评估则返回0时间
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重

    # return acc

if __name__ == "__main__":
    main(Config)

    # for model in ["bert"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索

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

    configs_to_test = []

    # 生成所有配置组合
    for model in ["gated_cnn", 'bert', 'lstm']:
        for lr in [1e-3, 1e-4]:
            for hidden_size in [128]:
                for batch_size in [64, 128]:
                    for pooling_style in ["avg", 'max']:
                        config = Config.copy()
                        config["model_type"] = model
                        config["learning_rate"] = lr
                        config["hidden_size"] = hidden_size
                        config["batch_size"] = batch_size
                        config["pooling_style"] = pooling_style
                        configs_to_test.append(config)

    # 运行所有配置
    for i, config in enumerate(configs_to_test):
        logger.info(f"开始测试配置 {i + 1}/{len(configs_to_test)}")
        logger.info(f"模型类型: {config['model_type']}, 学习率: {config['learning_rate']}, "
                    f"隐藏层大小: {config['hidden_size']}, 批大小: {config['batch_size']}, "
                    f"池化方式: {config['pooling_style']}")

        # 运行训练和评估
        acc, eval_time = main(config)

        # 记录结果
        results.append({
            "model": config["model_type"],
            "learning_rate": config["learning_rate"],
            "hidden_size": config["hidden_size"],
            "batch_size": config["batch_size"],
            "pooling_style": config["pooling_style"],
            "accuracy": acc,
            "evaluation_time": eval_time
        })

    # 打印结果表格
    print("\n模型性能对比：")
    headers = ["模型", "学习率", "隐藏层", "批大小", "池化方式", "准确率", "评估时间(s)"]
    table_data = []

    for res in results:
        table_data.append([
            res["model"],
            res["learning_rate"],
            res["hidden_size"],
            res["batch_size"],
            res["pooling_style"],
            f"{res['accuracy']:.4f}",
            f"{res['evaluation_time']:.2f}"
        ])

    # 按评估时间排序
    sorted_data = sorted(table_data, key=lambda x: float(x[-1]))

    print(tabulate(sorted_data, headers=headers, tablefmt="grid"))

    # 打印最佳配置
    best_config = max(results, key=lambda x: x["accuracy"])
    print("\n最佳配置：")
    print(f"模型类型: {best_config['model']}")
    print(f"学习率: {best_config['learning_rate']}")
    print(f"隐藏层大小: {best_config['hidden_size']}")
    print(f"批大小: {best_config['batch_size']}")
    print(f"池化方式: {best_config['pooling_style']}")
    print(f"准确率: {best_config['accuracy']:.4f}")
    print(f"评估时间: {best_config['evaluation_time']:.2f}秒")


