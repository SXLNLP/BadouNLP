# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
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
    "pretrain_model_path":r"E:\BaiduNetdiskDownload\八斗nlp\week6\bert-base-chinese\bert-base-chinese",
    "seed": 987,
    "train_data_percent": 0.8,
    "dataset_path": "E:\jianguo\我的坚果云\dl_coding\w7\job\文本分类练习.csv"
}

