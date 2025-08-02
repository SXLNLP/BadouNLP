# -*- coding: utf-8 -*-
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from tqdm import tqdm

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    # 创建模型保存目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    
    # 初始化模型
    model = TorchModel(config)
    
    # 检查是否使用 GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("使用GPU训练")
        model = model.cuda()
    
    # 优化器
    optimizer = choose_optimizer(config, model)
    
    # 评估器
    evaluator = Evaluator(config, model, logger)
    
    # 训练过程
    for epoch in range(config["epoch"]):
        model.train()
        logger.info("epoch %d begin" % (epoch + 1))
        train_loss = []

        # 使用 tqdm 包装训练数据，添加进度条
        progress_bar = tqdm(train_data, desc=f"Training Epoch {epoch + 1}", leave=True)

        for batch in progress_bar:
            optimizer.zero_grad()
            
            # 如果用 GPU，把 batch 搬过去
            if cuda_flag:
                for key in batch:
                    batch[key] = batch[key].cuda()
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            
            # 前向传播 + 反向传播 + 更新
            loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
            # tqdm 显示当前 loss
            progress_bar.set_postfix(loss=loss.item())
        
        # 每个 epoch 打印平均损失
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        
        # 每个 epoch 评估一次
        evaluator.eval(epoch + 1)
    
    # 保存最终模型
    model_path = os.path.join(config["model_path"], "final_model.pth")
    torch.save(model.state_dict(), model_path)
    
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
