# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "E:\\BaiduNetdiskDownload\\week7 文本分类问题\\文本分类练习.csv",
    "valid_data_path": r"E:\BaiduNetdiskDownload\week7 文本分类问题\文本分类练习.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 40,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"F:\bert\bert-chinese\bert-base-chinese",
    "seed": 987
}

