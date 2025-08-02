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
    "epoch": 1,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.6,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "enviroment": "product",
    "model_type": "lstm",
    "num_layers": 2,
    "pooling_style": "max",
}
