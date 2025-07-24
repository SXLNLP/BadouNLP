# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "data_path": r"C:\BaiduNetdiskDownload\八斗精品班\week7 文本分类问题\文本分类练习.csv",
    "test_size": 0.2,
    "vocab_path": "chars.txt",
    "model_type": "rnn",
    "max_length": 30,
    "class_num": 2,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style": "abg",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 987
}
