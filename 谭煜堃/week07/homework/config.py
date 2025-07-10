# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "./homework/output",
    "data_path": "./homework/文本分类练习.csv",
    "vocab_path":"./homework/chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "class_num": 2,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"average",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 987
}

