# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "triplet_margin": 0.2,
    "distance_norm": 2,     # 距离度量（2表示欧氏距离）
    "grad_clip": 1.0,
    "model_path": "model_output",
    "schema_path": "../week8 文本匹配问题/data/schema.json",
    "train_data_path": "../week8 文本匹配问题/data/train.json",
    "valid_data_path": "../week8 文本匹配问题/data/valid.json",
    "vocab_path":"../week8 文本匹配问题/chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 30,
    "hard_negative_ratio": 0.5,  # 初始困难负样本比例
    "mining_start_epoch": 3,     # 从第几个epoch开始使用困难负样本
    "batch_size": 64,            # 用于困难负样本挖掘的批量大小
    "pooling_style":"max",
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path":r"D:\PycharmProjects\AI学习预习\week6+语言模型和预训练\bert-base-chinese"
}