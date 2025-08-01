# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    # 基础配置
    "vocab_path": "vocab.txt",  # 词表文件路径
    "bert_model_path": "bert-base-chinese",  # huggingface模型名或本地目录
    "train_data_path": "sample_data.json",  # 训练数据路径
    "valid_data_path": "sample_data.json",  # 验证数据路径
    
    # SFT训练配置
    "model_name": "sft_news_model",  # 模型保存名称
    "save_dir": "./checkpoints",  # 模型保存目录
    "max_length": 512,  # 最大序列长度
    "batch_size": 4,  # 批次大小
    "lr": 2e-5,  # 学习率
    "epochs": 5,  # 训练轮数
    "warmup_steps": 100,  # 预热步数
    "gradient_accumulation_steps": 4,  # 梯度累积步数
    "max_grad_norm": 1.0,  # 梯度裁剪
    
    # 数据处理配置
    "input_max_length": 256,  # 输入最大长度
    "output_max_length": 128,  # 输出最大长度
    "prompt_template": "请根据以下新闻内容生成标题：\n内容：{content}\n标题：",  # 提示模板
    
    # 训练控制
    "save_steps": 500,  # 保存步数
    "eval_steps": 200,  # 评估步数
    "logging_steps": 50,  # 日志步数
    "eval_strategy": "steps",  # 评估策略
    "save_strategy": "steps",  # 保存策略
    
    # 其他配置
    "seed": 42,  # 随机种子
    "fp16": False,  # 是否使用混合精度训练
    "dataloader_num_workers": 0,  # 数据加载器工作进程数
}

