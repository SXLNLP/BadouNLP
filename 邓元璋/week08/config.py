# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "vocab_path": "chars.txt",
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train.json",
    "valid_data_path": "data/valid.json",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 500,  # 每轮训练中采样数量
    "positive_sample_rate": 0.5,  # 正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "margin": 0.1,  # 三元组损失的margin
    "model_type": "bert",  # 或 "bert"
}