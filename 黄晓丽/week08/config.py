# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train.json",
    "valid_data_path": "data/valid.json",
    "vocab_path":"chars.txt",
    "max_length": 20,
    "hidden_size": 256,
    "epoch": 10,
    "batch_size": 64,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "triplet_margin": 0.5,  # 三元组损失的边界值
    "positive_sample_rate": 1.0,  # 固定使用三元组采样
}
