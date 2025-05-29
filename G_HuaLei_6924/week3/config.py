# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output.pth",
    "train_data_path": "../data/train_tag_news.json",
    "valid_data_path": "../data/valid_tag_news.json",
    "vocab_path":"vocab.json",
    "model_type":"RNN",
    "max_length": 30,
    "hidden_size": 60,
    "embedding_dim": 30,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "sample_amount": 10000,
    "batch_size": 20,
    "evaluate_batch_size": 200,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\AI_study\八斗学院\录播课件\第六周\bert-base-chinese",
    "seed": 987,
    "num_classes": 6, # 类别数
    "vocab_char_type": "english", # 字符集类型 -- 英文字母
}