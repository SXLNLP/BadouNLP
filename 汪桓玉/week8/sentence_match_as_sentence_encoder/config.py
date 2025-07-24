# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json", 
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,
    "positive_sample_rate": 0.5,  # 在三元组模式下可能不需要
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "margin": 0.1,  # 新增：三元组损失的margin参数
    "use_triplet_loss": True,  # 新增：是否使用三元组损失
}