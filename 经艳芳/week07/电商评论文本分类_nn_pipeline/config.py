# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "data_path" : "文本分类练习.csv",
    "vocab_path":"chars.txt",  #用于非BERT模型
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path":r"E:\BaiduNetdiskDownload\bert-base-chinese\bert-base-chinese",
    "seed": 987,
    "test_split_ratio": 0.2,
    "class_num":2
}

