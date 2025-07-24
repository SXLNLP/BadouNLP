# -*- coding: utf-8 -*-

"""
电商评论分类配置参数
"""

Config = {
    "model_path": "output",
    "train_data_path": "./data/train.csv",
    "valid_data_path": "./data/valid.csv",
    "vocab_path": "./data/chars.txt",
    "model_type": "bert",  # 可选 bert/lstm/cnn
    "max_length": 100,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 64,
    "pooling_style": "max",  # avg/max
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": "./pretrain_models/bert-base-chinese",
    "seed": 987,
    "class_num": 2,  # 二分类
     "label_map": {"差评": 0, "好评": 1},
    # 新增列名配置
    "text_column": "review",  # 文本列名
    "label_column": "label",  # 标签列名
    "require_improvement": 2000  # 若超过这个step还没提升，提前结束训练
}
