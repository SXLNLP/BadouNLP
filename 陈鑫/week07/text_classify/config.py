# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "./data/train_set.csv",
    "valid_data_path": "./data/test_set.csv",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\pretrain_models\bert-base-chinese",
    "seed": 987,
    "stat_file": r"./output/stat_data.csv",     # 统计预测结果的输出文件
}
