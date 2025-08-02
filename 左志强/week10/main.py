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
from evaluate import Evaluator
from loader import load_data
from transformers import BertModel, BertConfig, BertForMaskedLM

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
BERT+Mask自回归语言模型训练主程序
"""

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

# 实现BERT的mask机制
def mask_tokens(inputs, tokenizer, config):
    """准备 masked tokens inputs/labels 用于 masked language modeling: 80% MASK, 10% 随机, 10% 保持原词."""
    labels = inputs.clone()
    # 我们只对非特殊token应用mask (tokenizer.pad_token_id通常为0)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    # 获取需要mask的位置（不包括特殊token）
    probability_matrix = torch.full(labels.shape, config["mlm_probability"])
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    # 确保pad token不会被mask
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    # 确定哪些token需要被mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 只计算masked tokens的loss
    
    # 80%的概率将token替换为[MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10%的概率将token替换为随机词
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    # 10%的概率保持原词不变
    
    return inputs, labels

def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 加载tokenizer和预训练模型
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    
    # 如果有预训练模型，则加载；否则初始化一个新的BERT模型
    if config["pretrained_model_path"]:
        model = BertForMaskedLM.from_pretrained(config["pretrained_model_path"])
        logger.info(f"加载预训练模型: {config['pretrained_model_path']}")
    else:
        # 配置BERT模型
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"]
        )
        model = BertForMaskedLM(bert_config)
        logger.info("初始化新的BERT模型")
    
    # 加载tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        config["pretrained_model_path"] if config["pretrained_model_path"] else 'bert-base-chinese',
        do_lower_case=config["do_lower_case"]
    )
    
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, logger)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, tokenizer, logger)
    
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            # 假设batch_data是文本ID序列
            inputs = batch_data[0]
            
            # 应用masking机制
            inputs, labels = mask_tokens(inputs, tokenizer, config)
            
            # 将数据移到GPU
            if cuda_flag:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            # 前向传播
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            
            train_loss.append(float(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 打印训练进度
            if index % 100 == 0:
                logger.info(f"Batch {index}/{len(train_data)}, Loss: {loss.item():.4f}")
        
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        
        # 评估模型
        evaluator.eval(epoch)
        
        # 保存模型
        if epoch % config["save_steps"] == 0:
            model_path = os.path.join(config["model_path"], f"epoch_{epoch}")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info(f"模型已保存至: {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(config["model_path"], "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"最终模型已保存至: {final_model_path}")
    return

if __name__ == "__main__":
    main(Config)
