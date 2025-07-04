# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "orign_data_path":r'D:\AI课程学习\第七周\week7 文本分类问题\week7 文本分类问题\文本分类练习.csv ',
    'divide_test_sample_size':100,
    "model_path": "output",
    "train_data_path": "../data/train_data.csv",
    "valid_data_path": "../data/valid__data.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\AI课程学习\bert-base-chinese",
    "seed": 987
}

