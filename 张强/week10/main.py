# -*- coding: utf-8 -*-
from typing import Any

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel,choose_optimizer
# from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # test_data = load_data(config["test_data_path"], config) if "test_data_path" in config else None
    #加载模型
    model = TorchModel(config).to(device)
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # # 标识是否使用gpu
    # cuda_flag = torch.cuda.is_available()
    # if cuda_flag:
    #     logger.info("gpu可以使用，迁移模型至gpu")
    #     model = model.cuda()
    #加载效果测试类
    # evaluator = Evaluator(config, model, logger) if test_data else None
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for batch in train_data:
            optimizer.zero_grad()
            batch = [t.to(device) for t in batch]
            input_ids, attention_mask, labels = batch

            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # 打印训练信息
        avg_loss = np.mean(train_loss)
        logger.info(f"Train loss: {avg_loss:.4f}")
        # 示例生成
        examples = [
            "让他在半年之前，就不能做出",
            "李慕站在山路上，深深的呼吸"
        ]
        for text in examples:
            generated = generate_sentence(text, model)
            logger.info(f"Example: {text} -> {generated}")

        # 保存模型
        if (epoch + 1) % config["save_interval"] == 0:
            model_path = os.path.join(config["model_path"], f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)

    return model


def generate_sentence(prompt: str, model: torch.nn.Module) -> str:
    tokenizer = model.tokenizer
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output_text = prompt  # 初始化时保留原始prompt（不带空格）

    with torch.no_grad():
        while len(output_text) <= 30:
            outputs = model(input_ids)
            logits = outputs[0, -1, :]

            # Top-K采样
            top_k = 50
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_token = indices[torch.multinomial(probs, num_samples=1)]

            # 关键修改：直接拼接字符（不通过tokenizer.decode中间步骤）
            next_char = tokenizer.convert_ids_to_tokens(next_token.item())
            next_char = next_char.replace("##", "")  # 处理子词
            output_text += next_char

            # 更新输入
            next_token = next_token.view(1, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # 最终清理（移除所有空格）
    return output_text.replace(" ", "")



if __name__ == "__main__":
    model = main(Config)
