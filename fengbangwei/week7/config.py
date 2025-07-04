# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "D:/AI/ai_project/deepseek/week7/data/train_tag_review.json",
    "valid_data_path": "D:/AI/ai_project/deepseek/week7/data/valid_tag_review.json",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 25,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\BaiduNetdiskDownload\AI\nlp\第六周 语言模型\bert-base-chinese",
    "seed": 987
}
