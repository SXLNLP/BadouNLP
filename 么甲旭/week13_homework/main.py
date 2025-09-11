# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import torch.nn as nn
import logging
from config import Config
from model import TorchModel
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

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
    dev_data = load_data(config["dev_data_path"], config)  # 添加验证数据加载

    # 初始化模型
    model = TorchModel(config)  # 实例化模型

    # 大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)  # 修改为TOKEN_CLS
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)  # 修改为TOKEN_CLS
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)  # 修改为TOKEN_CLS

    # 应用PEFT配置
    if tuning_tactics != "full_finetune":  # 添加完整微调选项
        model = get_peft_model(model, peft_config)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        eps=config["adam_epsilon"]
    )

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    best_f1 = 0.0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                # 将字典中的每个张量移到GPU
                batch_data = {k: v.cuda() for k, v in batch_data.items()}

            optimizer.zero_grad()

            # 从字典中获取数据
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # 获取logits

            # 调整形状计算损失
            batch_size, seq_len, num_labels = logits.shape
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(batch_size * seq_len, num_labels),
                labels.view(batch_size * seq_len)
            )

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch %d loss: %f" % (index, loss.item()))

        logger.info("epoch average loss: %f" % np.mean(train_loss))

        # 在验证集上评估
        metrics = evaluator.eval(dev_data)
        current_f1 = metrics["f1"]

        # 保存最佳模型
        if current_f1 > best_f1:
            best_f1 = current_f1
            model_path = os.path.join(config["model_path"], f"{tuning_tactics}_best.pth")
            save_tunable_parameters(model, model_path)
            logger.info(f"保存最佳模型: epoch {epoch}, f1: {best_f1:.4f}")

    logger.info("训练完成！最佳F1分数: %.4f" % best_f1)
    return best_f1


def save_tunable_parameters(model, path):
    """保存可训练的参数"""
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)
    logger.info(f"模型参数已保存到: {path}")


if __name__ == "__main__":
    main(Config)
