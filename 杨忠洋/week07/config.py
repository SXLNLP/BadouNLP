# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "raw_data_path":  "./data/文本分类练习.csv",
    "train_data_path": "./data/train_data.json",
    "valid_data_path": "./data/valid_data.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 50,  # 根据分析结果调整
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,  # 减少epoch防止过拟合
    "batch_size": 32,  # 减小batch size
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT需要更小的学习率
    "pretrain_model_path": r"D:\AI\八斗精品班\第六周 语言模型和预训练\bert-base-chinese",
    "seed": 987,
    "class_num": 2,  # 二分类
    "dropout_rate": 0.3,  # 新增dropout率
    "weight_decay": 0.01  # 新增权重衰减
}

