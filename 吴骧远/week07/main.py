# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
import time
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data, process_csv_to_json
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

    # 记录开始时间
    start_time = time.time()

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
        acc = evaluator.eval(epoch)

    # 计算训练时间
    training_time = time.time() - start_time

    # 速度测试
    valid_data = load_data(config["valid_data_path"], config, shuffle=False)
    model.eval()
    speed_start = time.time()
    sample_count = 0
    for batch_data in valid_data:
        if cuda_flag:
            batch_data = [d.cuda() for d in batch_data]
        input_ids, labels = batch_data
        with torch.no_grad():
            _ = model(input_ids)
        sample_count += len(input_ids)
        if sample_count >= 100:
            break
    speed_time = time.time() - speed_start
    samples_per_second = sample_count / speed_time

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, training_time, samples_per_second


if __name__ == "__main__":
    # 处理CSV文件
    csv_file = "文本分类练习.csv"
    if os.path.exists(csv_file) and not os.path.exists("data/train_tag_news.json"):
        logger.info("处理CSV文件...")
        process_csv_to_json(csv_file)

    # 存储结果
    results = []

    # 对比三个模型
    for model in ["cnn", "lstm", "bert"]:
        Config["model_type"] = model

        # 针对不同模型调整参数
        if model == "bert":
            Config["learning_rate"] = 5e-5
            Config["batch_size"] = 32
            Config["epoch"] = 5
        else:
            Config["learning_rate"] = 1e-3
            Config["batch_size"] = 64
            Config["epoch"] = 10

        logger.info(f"开始训练模型: {model}")
        acc, train_time, speed = main(Config)
        results.append({
            'model': model,
            'accuracy': acc,
            'train_time': train_time,
            'speed': speed
        })
        print(f"模型: {model}, 准确率: {acc:.4f}, 训练时间: {train_time:.2f}秒, 预测速度: {speed:.2f}样本/秒")

    # 输出对比结果
    print("\n" + "=" * 80)
    print("模型对比结果")
    print("=" * 80)
    print(f"{'模型':<10} {'准确率':<10} {'训练时间(秒)':<15} {'预测速度(样本/秒)':<20}")
    print("-" * 80)
    for result in results:
        print(
            f"{result['model']:<10} {result['accuracy']:<10.4f} {result['train_time']:<15.2f} {result['speed']:<20.2f}")
    print("=" * 80)