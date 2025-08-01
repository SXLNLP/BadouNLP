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
    train_data = load_data(config["data_path"], config)
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
        current_penalty = 1.0 + min(0.5, epoch * 0.02)
        for batch in train_data:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # 打印训练信息
        avg_loss = np.mean(train_loss)
        logger.info(f"Train loss: {avg_loss:.4f}")
        # 示例生成
        examples = [
            "韩国7岁女童被性侵 总统向国民致歉",
            "安徽“洋城管”称自己若事先知情不会参加"
        ]
        for text in examples:
            generated = generate_text(text,model)
            logger.info(f"生成结果: {text} → {generated}")

        # 保存模型
        if (epoch + 1) % config["save_interval"] == 0:
            model_path = os.path.join(config["model_path"], f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)

    return model
def generate_text(prompt: str, model: torch.nn.Module):
    tokenizer = model.tokenizer
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output_text = prompt  # 初始化时保留原始prompt（不带空格）
    seq_len = input_ids.shape[1]
    attention_mask = torch.tril(torch.ones(seq_len, seq_len,dtype=torch.long)).to(model.device).unsqueeze(0)

    with torch.no_grad():
        while len(output_text) <= 150:
            outputs = model(input_ids,attention_mask)
            logits = outputs[:, -1, :]

            # Top-K采样
            top_k = 50
            top_logits, top_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_logits, dim=-1)

            # 从top_k中采样一个token
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token_id = top_indices.gather(-1, next_token_idx)

            # 将新token添加到输入序列
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # 更新attention mask - 扩展为新的下三角矩阵
            new_len = input_ids.shape[1]
            # 创建新的下三角矩阵
            new_attention_mask = torch.tril(torch.ones(1, new_len, new_len, dtype=torch.long)).to(device)
            attention_mask = new_attention_mask

            # 将新token添加到输出文本
            new_token = tokenizer.convert_ids_to_tokens(next_token_id.item())
            new_token = new_token.replace("##", "")  # 处理子词
            output_text += new_token

            # 如果遇到结束符则停止
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # 最终清理（移除所有空格）
        return output_text.replace(" ", "")

# def generate_text(prompt: str, model: torch.nn.Module):
#     tokenizer = model.tokenizer
#     model.eval()
#     device = next(model.parameters()).device
#
#     input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
#     output_text = prompt  # 初始化时保留原始prompt（不带空格）
#
#     with torch.no_grad():
#         while len(output_text) <= 150:
#             outputs = model(input_ids)
#             logits = outputs[0, -1, :]
#
#             # Top-K采样
#             top_k = 50
#             values, indices = torch.topk(logits, top_k)
#             probs = torch.softmax(values, dim=-1)
#             next_token = indices[torch.multinomial(probs, num_samples=1)]
#
#             # 关键修改：直接拼接字符（不通过tokenizer.decode中间步骤）
#             next_char = tokenizer.convert_ids_to_tokens(next_token.item())
#             next_char = next_char.replace("##", "")  # 处理子词
#             output_text += next_char
#
#             # 更新输入
#             next_token = next_token.view(1, 1)
#             input_ids = torch.cat([input_ids, next_token], dim=-1)
#
#             if next_token.item() == tokenizer.eos_token_id:
#                 break
#
#     # 最终清理（移除所有空格）
#     return output_text.replace(" ", "")






if __name__ == "__main__":
    model = main(Config)
