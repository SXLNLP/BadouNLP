# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "/Library/workerspace/python_test/badou_demo1/week07homework/data/train_tag_news.json",
    "valid_data_path": "/Library/workerspace/python_test/badou_demo1/week07homework/data/valid_tag_news.json",
    "vocab_path":"/Library/workerspace/python_test/badou_demo1/week07homework/chars.txt",
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
    "pretrain_model_path":"/Library/workerspace/python_test/badou_demo1/week07homework/bert-base-chinese",
    "seed": 987
}

