# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "pretrain_model_path": r"/Users/juju/BaDou/bert-base-chinese",
    "model_path": "model_output",
    "schema_path": "../data/schema.json",  # target问题对应的index
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path": "../chars.txt",
    "max_length": 25,
    "hidden_size": 256,
    "epoch": 10,
    "batch_size": 64,
    "epoch_data_size": 200,  #每轮训练中采样数量
    "positive_sample_rate": 0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
