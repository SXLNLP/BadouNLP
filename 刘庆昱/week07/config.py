# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "dataset_path": "/Users/liuqingyu/aiStudio/py312/week07/data_set.csv",
    "train_data_path": "./data/train_data.txt",
    "valid_data_path": "./data/valid_data.txt",
    "vocab_path": "chars.txt",
    "model_type": "gated_cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 256,
    "pooling_style": "avg",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"/Users/liuqingyu/aiStudio/py312/week07/bert-base-chinese",
    "seed": 987
}
