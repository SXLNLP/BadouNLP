# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": r"D:\nlp516\week7 文本分类问题\homework\data\文本分类数据 - 训练.csv",
    "valid_data_path": r"D:\nlp516\week7 文本分类问题\homework\data\文本分类数据 - 评估.csv",
    "vocab_path":r"D:\nlp516\week7 文本分类问题\homework\week07_homework\chars.txt",
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
    "pretrain_model_path":r"D:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese\bert-base-chinese",
    "seed": 987
}

