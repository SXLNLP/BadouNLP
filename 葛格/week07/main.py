# -*- coding: utf-8 -*-

import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
主函数，最后生成的结果在result.txt文件中
本来想把所有模型一次性都训了，但是OOM了，因为只有1张显卡空闲
所以生成部分结果
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc, infer_time = evaluator.eval(epoch)

        # 从模型里获取真实的 hidden_size
        real_hidden_size = model.classify.in_features

    return acc, infer_time, real_hidden_size


if __name__ == "__main__":
    all_models = [
        # "fast_text",
        # "cnn",
        # "gated_cnn",
        "stack_gated_cnn",
        # "rnn",
        "lstm",
        # "gru",
        # "rcnn",
        "bert",
        # "bert_lstm",
        # "bert_cnn",
        # "bert_mid_layer"
    ]

    results = []
    for model in all_models:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        acc, infer_time , real_hidden_size = main(Config)  # 训练模型并获取准确率和预测时间
                        logger.info(f"[RESULT] 模型: {model}, 学习率: {lr}, 实际 hidden_size: {real_hidden_size}")
                        results.append({
                            "model": model,
                            "learning_rate": lr,
                            "hidden_size": real_hidden_size,
                            "batch_size": batch_size,
                            "pooling_style": pooling_style,
                            "acc": acc,
                            "time": infer_time
                        })

    print("\n对比结果：")
    # 格式化输出列标题
    print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<8} {:<10}".format(
        "model", "lr", "hidden_size", "batch_size", "pooling_style", "acc", "time(ms)"
    ))

    # 输出每个模型的结果
    for r in results:
        print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<8.4f} {:<10.4f}".format(
            r["model"], r["learning_rate"], r["hidden_size"], r["batch_size"], r["pooling_style"], r["acc"], r["time"]
        ))

    best_result = max(results, key=lambda x: x["acc"])
    print("最佳模型配置：", best_result)
