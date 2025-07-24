# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "train_text_classify.csv",
    "valid_data_path": "valid_text_classify.csv",
    "vocab_path":"chars.txt",
    "model_type":"cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"/Users/se7en/project/demo/example-py/cv/deepthink/week6 语言模型和预训练/下午/bert-base-chinese",
    "seed": 987
}

