# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import time
import json
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import logging

from config import Config
from sft_loader import create_dataloaders, set_seed
from sft_model import SFTModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sft_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SFTTrainer:
    """SFT训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        set_seed(config["seed"])
        
        # 创建保存目录
        os.makedirs(config["save_dir"], exist_ok=True)
        
        # 初始化模型
        self.model = SFTModel(config)
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.tokenizer = create_dataloaders(config)
        
        # 初始化优化器和调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=0.01
        )
        
        # 计算总步数
        total_steps = len(self.train_loader) * config["epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=total_steps
        )
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=f"{config['save_dir']}/logs")
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["max_grad_norm"]
            )
            
            # 优化器步进
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # 更新全局步数
            self.global_step += 1
            
            # 记录损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 记录到TensorBoard
            if self.global_step % self.config["logging_steps"] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            
            # 保存检查点
            if self.global_step % self.config["save_steps"] == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}")
            
            # 验证
            if self.global_step % self.config["eval_steps"] == 0:
                val_loss = self.validate()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model")
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = os.path.join(self.config["save_dir"], name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_path)
        
        # 保存训练状态
        state_dict = {
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(state_dict, os.path.join(checkpoint_path, 'training_state.pt'))
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        # 加载模型
        self.model = SFTModel.from_pretrained(checkpoint_path, self.config)
        self.model.to(self.device)
        
        # 加载训练状态
        state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location=self.device)
            self.global_step = state_dict['global_step']
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.best_val_loss = state_dict['best_val_loss']
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """开始训练"""
        logger.info("Starting SFT training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config["epochs"] + 1):
            logger.info(f"Epoch {epoch}/{self.config['epochs']}")
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存epoch检查点
            self.save_checkpoint(f"epoch_{epoch}")
        
        # 保存最终模型
        self.save_checkpoint("final_model")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        self.writer.close()

def main():
    """主函数"""
    # 创建训练器
    trainer = SFTTrainer(Config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 