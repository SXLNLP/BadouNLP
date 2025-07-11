# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output", #模型存放的目录
    # "train_data_path": "../data/train_tag_news.json",#训练数据集文件
    # "valid_data_path": "../data/valid_tag_news.json",#验证数据集文件
    "train_data_path": "../data/文本分类练习_train.csv",#训练数据集文件
    "valid_data_path": "../data/文本分类练习_test.csv",#验证数据集文件
    "vocab_path":"chars.txt",#词表文件
    "model_type":"bert",#模型类型
    "max_length": 30,#句子的最大长度，即字符数量
    "hidden_size": 256,#隐藏层维度大小
    "kernel_size": 3,#内核大小
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\www.root\bert-base-chinese",
    "seed": 987
}

