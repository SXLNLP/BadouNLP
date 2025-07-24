# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "E:/NLP学习/week8作业/data/schema.json",
    "train_data_path": "E:/NLP学习/week8作业/data/train.json",
    "valid_data_path": "E:/NLP学习/week8作业/data/valid.json",
    "vocab_path":"E:/NLP学习/week8作业/data/chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.66,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
