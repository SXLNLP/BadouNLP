# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "../model_output", #模型保存路径
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "triplet_margin": 0.1,      #三元組損失的邊距參數
    "optimizer": "adam",
    "learning_rate": 1e-3,
}