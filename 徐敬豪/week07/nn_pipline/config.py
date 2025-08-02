# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "E:/NLP学习/week7作业/新建文件夹/nn_pipline/train_data.txt",
    "valid_data_path": "E:/NLP学习/week7作业/新建文件夹/nn_pipline/valid_data.txt",
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
    "pretrain_model_path":r"E:\学习过程\week6\bert-base-chinese",
    "seed": 987
}

