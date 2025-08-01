# -*- coding: utf-8 -*-
import sys
import torch
import os
import random
import numpy as np
import time
import logging
import json
from config import Config
from sft_evaluate import Evaluator
from sft_loader import load_sft_data, create_sft_config

# 这个transformer是本文件夹下的代码，和transformers第三方库是两回事
from transformer.Models import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
SFT (Supervised Fine-tuning) 训练主程序
"""

def choose_optimizer(config, model):
    """选择优化器"""
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

def create_scheduler(optimizer, config):
    """创建学习率调度器"""
    if "warmup_steps" in config and config["warmup_steps"] > 0:
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < config["warmup_steps"]:
                return step / config["warmup_steps"]
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    return None

class SFTLoss(torch.nn.Module):
    """SFT专用损失函数"""
    def __init__(self, ignore_index=0, response_loss_weight=1.0):
        super(SFTLoss, self).__init__()
        self.ignore_index = ignore_index
        self.response_loss_weight = response_loss_weight
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, pred, gold):
        """
        计算SFT损失，只对response tokens计算损失
        """
        # 这里简化处理，对所有tokens计算损失
        # 在实际应用中，可以通过mask来控制只对response部分计算损失
        loss = self.loss_func(pred, gold.view(-1))
        return loss * self.response_loss_weight

def main_sft(config):
    """SFT训练主函数"""
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 打印配置信息
    logger.info("SFT Training Configuration:")
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    
    # 加载模型
    model = Transformer(
        config["vocab_size"], config["vocab_size"], 0, 0,
        d_word_vec=128, d_model=128, d_inner=256,
        n_layers=1, n_head=2, d_k=64, d_v=64,
    )
    
    # 如果有预训练模型，可以加载
    if "pretrained_model_path" in config and os.path.exists(config["pretrained_model_path"]):
        logger.info(f"Loading pretrained model from {config['pretrained_model_path']}")
        model.load_state_dict(torch.load(config["pretrained_model_path"], map_location='cpu'))
    
    # GPU设备检查
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 学习率调度器
    scheduler = create_scheduler(optimizer, config)
    
    # 加载SFT训练数据
    train_data = load_sft_data(config["train_data_path"], config, logger)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 加载SFT损失函数
    loss_func = SFTLoss(
        ignore_index=config["pad_idx"],
        response_loss_weight=config.get("response_loss_weight", 1.0)
    )
    
    logger.info("开始SFT训练...")
    
    # SFT训练循环
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        
        logger.info(f"SFT Epoch {epoch} begin")
        train_loss = []
        
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            input_seq, target_seq, gold = batch_data
            
            # 前向传播
            pred = model(input_seq, target_seq)
            loss = loss_func(pred, gold)
            
            # 记录损失
            train_loss.append(float(loss))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            if "max_grad_norm" in config:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            global_step += 1
            
            # 定期打印训练进度
            if index % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}, Step {index}, Loss: {loss:.4f}, LR: {current_lr:.2e}")
        
        # 每个epoch结束后的统计
        avg_loss = np.mean(train_loss)
        logger.info(f"SFT Epoch {epoch} average loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(config["model_path"], "best_sft_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"保存最佳SFT模型到: {best_model_path}")
        
        # 定期评估
        if epoch % 5 == 0:
            evaluator.eval(epoch)
        
        # 保存检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config["model_path"], f"sft_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"保存检查点到: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(config["model_path"], "final_sft_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"SFT训练完成，最终模型保存到: {final_model_path}")
    
    return model

def main():
    """主函数"""
    # 创建SFT配置
    sft_config = create_sft_config(Config)
    
    # 添加一些SFT特有的配置
    sft_config.update({
        "max_grad_norm": 1.0,              # 梯度裁剪
        "warmup_steps": 100,               # 预热步数
        "learning_rate": 1e-4,             # 较小的学习率
        "epoch": 30,                       # 较少的训练轮数
        "response_loss_weight": 1.0,       # 回答部分损失权重
    })
    
    # 如果有之前训练的模型，可以指定预训练模型路径
    if os.path.exists("output/epoch_200.pth"):
        sft_config["pretrained_model_path"] = "output/epoch_200.pth"
        logger.info("发现预训练模型，将基于该模型进行SFT训练")
    
    # 开始SFT训练
    model = main_sft(sft_config)
    
    logger.info("SFT训练任务完成！")
    
    return model

if __name__ == "__main__":
    main()