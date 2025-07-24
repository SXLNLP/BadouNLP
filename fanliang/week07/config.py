# -*- coding: utf-8 -*-
import os
"""
配置参数信息
"""

Config_news = {
    "model_path": "output",
    "train_data_path": os.path.dirname(os.path.abspath(__file__))+"/../data/train_tag_news.json",
    "valid_data_path": os.path.dirname(os.path.abspath(__file__))+"/../data/valid_tag_news.json",
    "vocab_path":os.path.dirname(os.path.abspath(__file__))+"/chars.txt",
    "model_type":"bert",
    "max_length": 30,#30
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,#15
    "batch_size": 10, # 128
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":os.getcwd()+"/models/bert-base-chinese",
    "seed": 987
}

Config = {
    "model_path": "output", #机器配置太低，选择test小数据量，真实数据是restaurant_remark_train.csv和restaurant_remark_valid.csv
    "train_data_path": os.path.dirname(os.path.abspath(__file__))+"/../data/restaurant_remark_test.csv",
    "valid_data_path": os.path.dirname(os.path.abspath(__file__))+"/../data/restaurant_remark_test.csv",
    "vocab_path":os.path.dirname(os.path.abspath(__file__))+"/chars.txt",
    "model_type":"bert",
    "max_length": 10,#30
    "hidden_size": 32,#256
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,#15
    "batch_size": 5, # 128
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":os.getcwd()+"/models/bert-base-chinese",
    "seed": 987
}