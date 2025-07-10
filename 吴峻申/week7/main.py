# -*- coding: utf-8 -*-
import logging
import os
import random
import numpy as np
import torch
import pandas as pd
import time
from config import Config
from evaluate import Evaluator
from loader import load_data
from model import TorchModel, choose_optimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 全局变量存储性能结果
performance_results = []


def test_predict_speed(model, data_loader, num_samples=100):
    """测试模型预测100条数据的速度"""
    model.eval()
    total_time = 0
    samples_processed = 0

    with torch.no_grad():
        start_time = time.time()

        for batch_data in data_loader:
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            input_ids, _ = batch_data
            _ = model(input_ids)  # 进行预测

            batch_size = input_ids.size(0)
            samples_processed += batch_size

            if samples_processed >= num_samples:
                break

        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # 转换为毫秒

    # 计算预测100条样本的平均时间
    time_per_100 = total_time * (100 / samples_processed)
    return round(time_per_100, 2)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 创建当前配置的性能记录
    current_performance = {
        "Model": config["model_type"],
        "Learning_Rate": config["learning_rate"],
        "Hidden_Size": config["hidden_size"],
        "acc": 0,
        "time(ms)": 0
    }

    try:
        logger.info(
            f"开始训练模型: {config['model_type']}, LR: {config['learning_rate']}, Hidden: {config['hidden_size']}")

        train_data = load_data(config["train_data_path"], config)
        valid_data = load_data(config["valid_data_path"], config, shuffle=False)

        # 创建用于速度测试的数据加载器
        speed_test_data = load_data(config["valid_data_path"], config, shuffle=False)

        model = TorchModel(config)
        cuda_flag = torch.cuda.is_available()

        if cuda_flag:
            logger.info("使用GPU加速")
            model = model.cuda()

        optimizer = choose_optimizer(config, model)
        evaluator = Evaluator(config, model, logger)

        best_acc = 0
        for epoch in range(config["epoch"]):
            epoch += 1
            model.train()
            logger.info(f"第{epoch}轮训练开始")
            train_loss = []

            for index, batch_data in enumerate(train_data):
                if cuda_flag:
                    batch_data = [d.cuda() for d in batch_data]

                optimizer.zero_grad()
                input_ids, labels = batch_data
                loss = model(input_ids, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

                if index % 10 == 0:
                    logger.info(f"批次 {index}/{len(train_data)} 损失: {loss.item():.4f}")

            logger.info(f"本轮平均损失: {np.mean(train_loss):.4f}")
            acc = evaluator.eval(valid_data, epoch)

            if acc > best_acc:
                best_acc = acc
                model_path = os.path.join(config["model_path"], f"best_model_{config['model_type']}.pth")
                torch.save(model.state_dict(), model_path)
                logger.info(f"保存最佳模型，准确率: {acc:.4f}")

        # 记录最佳准确率
        current_performance["acc"] = round(best_acc, 4)
        logger.info(f"最终准确率: {best_acc:.4f}")

        # 测试预测速度
        predict_time = test_predict_speed(model, speed_test_data)
        current_performance["time(ms)"] = predict_time
        logger.info(f"预测100条数据耗时: {predict_time} ms")

        # 添加到性能结果列表
        performance_results.append(current_performance)

        return best_acc

    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        current_performance["acc"] = 0
        current_performance["time(ms)"] = 0
        performance_results.append(current_performance)
        return 0


def generate_performance_report():
    """生成性能报告并保存为CSV文件"""
    if not performance_results:
        logger.warning("没有性能数据可生成报告")
        return None

    # 创建DataFrame
    df = pd.DataFrame(performance_results)

    # 设置表头
    columns = ["Model", "Learning_Rate", "Hidden_Size", "acc", "time(ms)"]

    # 生成报告文件路径
    report_path = os.path.join(Config["model_path"], "model_performance_report.csv")

    # 保存为CSV
    df.to_csv(report_path, index=False)
    logger.info(f"模型性能报告已保存至: {report_path}")

    # 打印报告
    logger.info("\n模型性能汇总报告:")
    logger.info(df.to_string(index=False))

    return df


if __name__ == "__main__":
    # 定义要测试的模型配置组合
    config_combinations = [
        {"model_type": "rnn", "learning_rate": 1e-3, "hidden_size": 128},
        {"model_type": "rnn", "learning_rate": 1e-4, "hidden_size": 256},
        {"model_type": "lstm", "learning_rate": 1e-3, "hidden_size": 128},
        {"model_type": "lstm", "learning_rate": 1e-4, "hidden_size": 256},
        {"model_type": "gated_cnn", "learning_rate": 1e-3, "hidden_size": 128},
        {"model_type": "gated_cnn", "learning_rate": 1e-4, "hidden_size": 256},
        {"model_type": "bert", "learning_rate": 5e-5, "hidden_size": 768},
    ]

    # 备份原始配置
    original_config = Config.copy()

    # 遍历所有配置组合
    for config_params in config_combinations:
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"开始测试配置: 模型={config_params['model_type']}, 学习率={config_params['learning_rate']}, 隐藏层={config_params['hidden_size']}")
        logger.info(f"{'=' * 60}")

        # 更新当前配置
        current_config = original_config.copy()
        current_config.update(config_params)

        # 运行训练和评估
        main(current_config)

    # 生成最终性能报告
    generate_performance_report()