# -*- coding: utf-8 -*-
import torch
import os
import random
import numpy as np
import logging
import time
import pandas as pd
import psutil
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 设置随机种子确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 模型训练和评估函数
def train_and_evaluate(config):
    """训练模型并返回评估指标"""
    results = {}
    try:
        # 开始计时和内存监控
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / 1024 ** 2  # 起始内存(MB)

        logger.info(f"\n 开始实验: 模型={config['model_type']}, "
                    f"lr={config['learning_rate']}, hidden={config['hidden_size']}, "
                    f"batch={config['batch_size']}, pooling={config['pooling_style']}")

        # 创建模型目录
        os.makedirs(config["model_path"], exist_ok=True)

        # 加载数据
        train_data = load_data(config["train_data_path"], config, shuffle=True)
        valid_data = load_data(config["valid_data_path"], config, shuffle=False)

        # 初始化模型
        model = TorchModel(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 设置优化器
        optimizer = choose_optimizer(config, model)
        evaluator = Evaluator(config, model, logger)

        # 训练循环
        train_losses = []
        for epoch in range(config["epoch"]):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0

            for batch_data in train_data:
                # 迁移数据到设备
                input_ids = batch_data[0].to(device)
                labels = batch_data[1].to(device)

                optimizer.zero_grad()
                loss = model(input_ids, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 计算平均损失
            avg_loss = epoch_loss / len(train_data)
            train_losses.append(avg_loss)
            epoch_time = time.time() - epoch_start

            # 验证评估
            acc = evaluator.eval(epoch + 1)
            logger.info(f"Epoch {epoch + 1}/{config['epoch']} | Loss: {avg_loss:.4f} | "
                        f"Valid Acc: {acc:.4f} | Time: {epoch_time:.2f}s")

        # 计算总资源消耗
        total_time = time.time() - start_time
        mem_end = process.memory_info().rss / 1024 ** 2
        mem_used = mem_end - mem_start

        # 计算模型大小
        param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size = param_size * 4 / (1024 ** 2)  # 转换为MB

        # 收集结果
        results = {
            "model": config["model_type"],
            "learning_rate": config["learning_rate"],
            "hidden_size": config["hidden_size"],
            "batch_size": config["batch_size"],
            "pooling_style": config["pooling_style"],
            "train_loss": train_losses[-1],  # 最终损失
            "valid_acc": acc,  # 最终准确率
            "total_time": total_time,  # 总耗时(秒)
            "memory_used": mem_used,  # 内存增量(MB)
            "model_size": model_size,  # 模型大小(MB)
        }

        logger.info(f" 实验完成 | 准确率: {acc:.4f} | 耗时: {total_time:.2f}s | "
                    f"内存占用: {mem_used:.2f}MB | 模型大小: {model_size:.2f}MB")

    except Exception as e:
        logger.error(f" 实验失败: {str(e)}")
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    set_seed(Config["seed"])

    # 结果收集器
    all_results = []

    # 定义实验参数组合
    model_types = ["fast_text", "lstm", "gru", "cnn", "gated_cnn", "bert", "bert_lstm"]
    learning_rates = [1e-3, 1e-4, 1e-5]
    hidden_sizes = [128, 256]
    batch_sizes = [16, 32]
    pooling_styles = ["avg", "max"]

    # 网格搜索实验
    for model in model_types:
        for lr in learning_rates:
            for hidden in hidden_sizes:
                for batch in batch_sizes:
                    for pooling in pooling_styles:
                        # 创建实验配置副本
                        exp_config = Config.copy()
                        exp_config.update({
                            "model_type": model,
                            "learning_rate": lr,
                            "hidden_size": hidden,
                            "batch_size": batch,
                            "pooling_style": pooling,
                        })

                        # 运行实验
                        result = train_and_evaluate(exp_config)
                        all_results.append(result)

    # 创建结果数据框
    df_results = pd.DataFrame(all_results)

    # 计算额外指标
    df_results["time_per_epoch"] = df_results["total_time"] / Config["epoch"]
    df_results["score_per_time"] = df_results["valid_acc"] / df_results["total_time"]

    # 保存到Excel
    df_results.to_excel("model_comparison_results.xlsx", index=False)

    # 最佳模型分析
    best_acc = df_results["valid_acc"].max()
    best_time = df_results.loc[df_results["valid_acc"] >= best_acc * 0.99, "total_time"].min()
    best_model = df_results.loc[
        (df_results["valid_acc"] >= best_acc * 0.99) &
        (df_results["total_time"] == best_time),
        "model"
    ].values[0]

    logger.info(f"\n{'=' * 50}")
    logger.info(f" 最佳模型: {best_model} | 准确率: {best_acc:.4f}")
    logger.info(f" 完整结果已保存至 model_comparison_results.xlsx")
    logger.info(f"️ 最快高质量模型: {best_model} (准确率>{best_acc * 0.99:.4f}, 耗时{best_time:.2f}s)")
    logger.info(f"{'=' * 50}")

    print("\n实验结果摘要:")
    print(df_results[["model", "valid_acc", "total_time", "memory_used", "model_size"]])
