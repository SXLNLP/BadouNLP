# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    # "train_data_path": "./data/train_tag_news.json",
    "train_data_path": "./data/train_review.csv",
    "valid_data_path": "./data/valid_review.csv",
    # "valid_data_path": "./data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"rcnn",
    "max_length": 30,
    "hidden_size": 512,
    "kernel_size": 3,
    "num_layers": 10,
    "epoch": 20,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\BaDouCourse\PyTorchDemo\VScodeDemoProject\nn_pipline\bert-base-chinese",
    "seed": 987
}