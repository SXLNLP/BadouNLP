# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "entire_data_path": "data/customer_reviews.csv",
    "train_data_path": "data/customer_reviews_train.csv",
    "valid_data_path": "data/customer_reviews_test.csv",
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
    "dropout": 0.5,
    "log_level": "DEBUG", # or "INFO"
    # "pretrain_model_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese",
    "pretrain_model_path":"bert-base-chinese",
    "seed": 987
}

