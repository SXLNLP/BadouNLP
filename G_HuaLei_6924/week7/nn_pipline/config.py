# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    # "train_data_path": "../data/train_tag_news.json",
    # "valid_data_path": "../data/valid_tag_news.json",
    "train_data_path": "../week7_text_csv/train_consumption_review.json",
    "valid_data_path": "../week7_text_csv/valid_consumption_review.json",
    "vocab_path":"chars.txt",
    # "model_type":"cnn",
    "model_type":"cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 5,
    "num_layers": 4,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\AI_study\八斗学院\录播课件\第六周\bert-base-chinese",
    "seed": 987
}