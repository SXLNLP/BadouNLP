# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"N:\八斗\上一期\第七周 文本分类\week7 文本分类问题\homework\output",
    "train_data_path": r"N:\八斗\上一期\第七周 文本分类\week7 文本分类问题\homework\train_data.csv",
    "valid_data_path": r"N:\八斗\上一期\第七周 文本分类\week7 文本分类问题\homework\test_data.csv",
    "vocab_path":r"N:\八斗\上一期\第七周 文本分类\week7 文本分类问题\homework\chars.txt",
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
    "pretrain_model_path":r"N:\八斗\上一期\第六周 语言模型\bert-base-chinese",
    "seed": 987
}

