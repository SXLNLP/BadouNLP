# -*- coding: utf-8 -*-
import sys
import torch
import os
import logging
import json
from config import Config
from evaluate import Evaluator
from loader import load_data
from transformers import BertForMaskedLM, AdamW

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 加载BERT模型
    model = BertForMaskedLM.from_pretrained(config["bert_model_path"])
    model.to(config["device"])
    
    # 加载优化器
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, logger)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        
        for index, batch_data in enumerate(train_data):
            input_ids, attention_mask, target_ids = [x.to(config["device"]) for x in batch_data]
            
            # 创建masked_lm_labels - 使用目标序列作为标签
            labels = target_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100  # 忽略pad token
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss.append(loss.item())
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if index % 100 == 0:
                logger.info(f"Batch {index}, Loss: {loss.item():.4f}")
        
        avg_loss = sum(train_loss) / len(train_loss)
        logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        # 评估模型
        evaluator.eval(epoch)
        
        # 保存模型
        model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
    
    return

if __name__ == "__main__":
    main(Config)
