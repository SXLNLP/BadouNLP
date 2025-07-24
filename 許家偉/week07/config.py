# -*- coding: utf-8 -*-

"""
配置参数信息
注意的是，training_raw.csv和bert-base-chinese都不在這次的代碼上傳中
"""

Config = {
    "model_path": "output",
    "train_data_path": "./training_raw.csv", # 假設training_raw.csv在當前目錄
    "valid_data_path": "./training_raw.csv", # 假設training_raw.csv在當前目錄
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
    "pretrain_model_path":r"../bert-base-chinese", # 假設bert-base-chinese在當前目錄的上一層
    "seed": 987
}

